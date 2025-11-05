"""Generate CAA and STA steering vectors for AxBench Concept500 concepts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from axbench_utils import (
    AXBENCH_REPO_ID,
    build_contrastive_pairs,
    build_generation_dataset,
    ensure_directory,
    load_axbench_split,
    sanitize_model_name,
)
from steer.vector_generators.caa.generate_caa_hparam import CAAHyperParams
from steer.vector_generators.caa.generate_caa_vectors import generate_caa_vectors
from steer.vector_generators.sta.generate_sta_hparam import STAHyperParams
from steer.vector_generators.sta.generate_sta_vectors import generate_sta_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAA/STA steering vectors for AxBench Concept500."
    )
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["caa", "sta"],
        choices=["caa", "sta"],
        help="Which steering vectors to generate.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[20],
        help="Transformer layers to target (single layer for STA).",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="google/gemma-2-9b-it",
        help="Model to edit.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_root",
        default="vectors",
        help="Base directory where vectors will be stored.",
    )
    parser.add_argument(
        "--sta_trim",
        type=float,
        default=0.65,
        help="Trim value for STA steering.",
    )
    parser.add_argument(
        "--sta_mode",
        default="act_and_freq",
        choices=["act_and_freq", "only_act", "only_freq"],
    )
    parser.add_argument(
        "--sta_release",
        default="google/gemma-scope-9b-pt-res-canonical",
        help="SAE release to use for STA (supports canonical releases).",
    )
    parser.add_argument(
        "--sta_width",
        default="width_16k",
        help="SAE width bucket (e.g., width_16k, width_131k).",
    )
    parser.add_argument(
        "--sta_variant",
        default="canonical",
        help="SAE variant suffix (default: canonical).",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Falls back to cached login if omitted.",
    )
    return parser.parse_args()


def _build_sta_paths(args: argparse.Namespace) -> List[str]:
    if "sta" not in args.methods:
        return []
    if len(args.layers) != 1:
        raise ValueError(
            "STA generation currently supports exactly one layer at a time. "
            "Pass a single --layers value when generating STA vectors."
        )
    layer = args.layers[0]
    return [
        f"{args.sta_release}:layer_{layer}/{args.sta_width}/{args.sta_variant}"
    ]


def _store_metadata(output_dir: str, dataset_name: str, metadata: Dict) -> None:
    meta_path = os.path.join(output_dir, dataset_name, "metadata.json")
    ensure_directory(os.path.dirname(meta_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _store_pairs(output_dir: str, dataset_name: str, pairs: List[Dict]) -> None:
    pairs_path = os.path.join(output_dir, dataset_name, "contrastive_pairs.json")
    ensure_directory(os.path.dirname(pairs_path))
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)


def generate_caa(
    args: argparse.Namespace,
    dataset_name: str,
    generation_dataset: List[Dict],
    output_dir: str,
):
    caa_hparams = CAAHyperParams.from_hparams(
        "hparams/Steer/caa_hparams/generate_caa.yaml"
    )
    caa_hparams.model_name_or_path = args.model_name_or_path
    caa_hparams.device = args.device
    caa_hparams.torch_dtype = args.torch_dtype
    caa_hparams.layers = args.layers
    caa_hparams.seed = args.seed
    caa_hparams.steer_vector_output_dir = output_dir
    caa_hparams.steer_train_dataset = dataset_name

    print("\n=== Generating CAA vectors ===")
    generate_caa_vectors(
        caa_hparams,
        dataset=generation_dataset,
        dataset_name=dataset_name,
    )


def generate_sta(
    args: argparse.Namespace,
    dataset_name: str,
    generation_dataset: List[Dict],
    output_dir: str,
):
    sta_paths = _build_sta_paths(args)
    sta_hparams = STAHyperParams.from_hparams(
        "hparams/Steer/sta_hparams/generate_sta.yaml"
    )
    sta_hparams.model_name_or_path = args.model_name_or_path
    sta_hparams.device = args.device
    sta_hparams.torch_dtype = args.torch_dtype
    sta_hparams.layers = args.layers
    sta_hparams.sae_paths = sta_paths
    sta_hparams.trims = [args.sta_trim]
    sta_hparams.mode = args.sta_mode
    sta_hparams.seed = args.seed
    sta_hparams.steer_vector_output_dir = output_dir
    sta_hparams.steer_train_dataset = dataset_name

    print("\n=== Generating STA vectors ===")
    generate_sta_vectors(
        sta_hparams,
        dataset=generation_dataset,
        dataset_name=dataset_name,
    )


def main() -> None:
    args = parse_args()

    print("Loading AxBench Concept500 train split ...")
    train_records = load_axbench_split("train", token=args.hf_token)
    pairs, metadata = build_contrastive_pairs(train_records, args.concept_id)

    print(
        f"Found {metadata['pair_count']} contrastive pairs from "
        f"{metadata['positive_examples']} positive samples."
    )

    generation_dataset = build_generation_dataset(pairs)
    dataset_name = f"axbench_concept_{args.concept_id}"
    model_dir = os.path.join(
        args.output_root,
        sanitize_model_name(args.model_name_or_path),
    )
    ensure_directory(model_dir)

    _store_metadata(model_dir, dataset_name, metadata)
    _store_pairs(model_dir, dataset_name, pairs)

    if "caa" in args.methods:
        generate_caa(args, dataset_name, generation_dataset, model_dir)

    if "sta" in args.methods:
        generate_sta(args, dataset_name, generation_dataset, model_dir)

    print("\nCompleted vector generation.")
    print(f"Vectors saved under: {os.path.join(model_dir, dataset_name)}")


if __name__ == "__main__":
    main()

