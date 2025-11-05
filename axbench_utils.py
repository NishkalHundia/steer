"""Utility helpers for working with the AxBench Concept500 dataset."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


AXBENCH_REPO_ID = "pyvene/axbench-concept500"


def sanitize_model_name(model_name: str) -> str:
    """Replace path separators so model names can be used in directory names."""

    return model_name.replace("/", "-")


def load_axbench_split(
    split: str,
    token: Optional[str] = None,
    subdir: str = "",
) -> List[Dict]:
    """Load an AxBench split and return the records as a list of dictionaries.

    The dataset is stored as Parquet files on the Hub.  We fetch the Parquet file
    using ``hf_hub_download`` to avoid schema issues with ``datasets.load_dataset``.

    Args:
        split: ``"train"`` or ``"test"``.
        token: Optional Hugging Face token.  ``hf_hub_download`` also respects the
            token configured via ``huggingface-cli login`` so this argument is
            optional.

    Raises:
        RuntimeError: If the download fails (for example due to missing token).

    Returns:
        List of dictionaries representing the rows.
    """

    prefix = f"{subdir.strip('/')}/" if subdir else ""
    filename = f"{prefix}{split}/data.parquet"
    try:
        parquet_path = hf_hub_download(
            repo_id=AXBENCH_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:  # pragma: no cover - network failures
        raise RuntimeError(
            "Failed to download AxBench split from Hugging Face. "
            "Ensure you have access to the dataset and are logged in via "
            "`huggingface-cli login` or have `HF_TOKEN` set."
        ) from exc

    table = pq.read_table(parquet_path)
    return table.to_pylist()


def _infer_label(row: Dict) -> Optional[str]:
    """Infer whether a row is positive or negative.

    The dataset provides slightly different columns across splits.  We inspect the
    available columns and look for positive/negative cues.
    """

    priority_columns = [
        "dataset_category",
        "category",
        "label",
        "split_category",
    ]

    for key in priority_columns:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        # Handle boolean/integer labels
        if isinstance(value, bool):
            return "positive" if value else "negative"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return "positive"
            if value == 0:
                return "negative"
        value_str = str(value).strip().lower()
        if "positive" in value_str or value_str in {"pos", "match"}:
            return "positive"
        if "negative" in value_str or value_str in {"neg", "mismatch"}:
            return "negative"
    return None


def build_contrastive_pairs(
    train_records: Sequence[Dict],
    concept_id: int,
) -> Tuple[List[Dict], Dict]:
    """Create contrastive prompt pairs for a concept.

    The instructions require:

    * Positive examples: rows whose ``concept_id`` matches ``concept_id``.
    * Negative examples: rows with the same ``concept_genre`` *and* identical
      ``input`` prompt but labelled as negative.

    Args:
        train_records: Rows from the AxBench train split.
        concept_id: Target concept id.

    Returns:
        A tuple ``(pairs, metadata)`` where ``pairs`` is a list of dictionaries
        (question/matching/not_matching) and ``metadata`` contains information
        about the concept (genre, counts, etc.).

    Raises:
        ValueError: If no positive examples or no contrastive pairs are found.
    """

    positives: Dict[str, Dict] = {}
    genre: Optional[str] = None

    for row in train_records:
        if int(row.get("concept_id", -1)) != concept_id:
            continue
        label = _infer_label(row)
        if label != "positive":
            continue
        prompt = row.get("input")
        output = row.get("output")
        if not prompt or output is None:
            continue
        positives[prompt] = row
        genre = row.get("concept_genre", genre)

    if not positives:
        raise ValueError(
            f"No positive training examples found for concept_id={concept_id}."
        )

    if genre is None:
        raise ValueError(
            f"Unable to determine concept_genre for concept_id={concept_id}."
        )

    negatives: Dict[str, Dict] = {}

    for row in train_records:
        label = _infer_label(row)
        if label != "negative":
            continue
        if row.get("concept_genre") != genre:
            continue
        prompt = row.get("input")
        if prompt not in positives:
            continue
        if prompt not in negatives:
            negatives[prompt] = row

    pairs: List[Dict] = []
    for prompt, pos_row in positives.items():
        neg_row = negatives.get(prompt)
        if not neg_row:
            continue
        pairs.append(
            {
                "question": prompt,
                "matching": pos_row.get("output", ""),
                "not_matching": neg_row.get("output", ""),
                "metadata": {
                    "concept_id": concept_id,
                    "concept_genre": genre,
                    "output_concept": pos_row.get("output_concept"),
                },
            }
        )

    if not pairs:
        raise ValueError(
            "No contrastive pairs found. Ensure the dataset contains negative "
            "examples that share prompts with the positive examples."
        )

    metadata = {
        "concept_id": concept_id,
        "concept_genre": genre,
        "positive_examples": len(positives),
        "pair_count": len(pairs),
    }

    return pairs, metadata


def build_generation_dataset(pairs: Sequence[Dict]) -> List[Dict]:
    """Convert contrastive pairs to the format expected by EasyEdit generators."""

    dataset = []
    for pair in pairs:
        dataset.append(
            {
                "question": pair["question"],
                "matching": pair["matching"],
                "not_matching": pair["not_matching"],
            }
        )
    return dataset


def build_test_dataset(
    test_records: Sequence[Dict],
    concept_id: int,
) -> List[Dict]:
    """Create evaluation dataset containing prompts and expected outputs."""

    evaluation_items: List[Dict] = []
    for row in test_records:
        if int(row.get("concept_id", -1)) != concept_id:
            continue
        prompt = row.get("input")
        output = row.get("output")
        if not prompt:
            continue
        evaluation_items.append(
            {
                "input": prompt,
                "reference_response": output,
                "concept_genre": row.get("concept_genre"),
                "category": row.get("category"),
                "dataset_category": row.get("dataset_category"),
                "output_concept": row.get("output_concept"),
            }
        )

    if not evaluation_items:
        raise ValueError(
            f"No evaluation samples found for concept_id={concept_id} in the test split."
        )

    return evaluation_items


def build_contrastive_eval_sets(
    pairs: Sequence[Dict],
) -> Dict[str, List[Dict]]:
    """Create datasets for sanity checking using the training prompts.

    Returns a dictionary with two keys (``positive`` and ``negative``) so that
    the applier can evaluate both expectations separately.
    """

    positive_items: List[Dict] = []
    negative_items: List[Dict] = []

    for pair in pairs:
        prompt = pair["question"]
        positive_items.append(
            {
                "input": prompt,
                "reference_response": pair["matching"],
                "target_type": "positive",
            }
        )
        negative_items.append(
            {
                "input": prompt,
                "reference_response": pair["not_matching"],
                "target_type": "negative",
            }
        )

    return {
        "contrastive_positive": positive_items,
        "contrastive_negative": negative_items,
    }


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)

