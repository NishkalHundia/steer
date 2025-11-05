"""Sanity-check steering using the contrastive prompts from training."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from omegaconf import OmegaConf

from axbench_utils import (
    build_contrastive_eval_sets,
    build_contrastive_pairs,
    ensure_directory,
    load_axbench_split,
    sanitize_model_name,
)
from steer.vector_appliers.vector_applier import BaseVectorApplier


METHOD_CONFIGS = {
    "caa": "hparams/Steer/caa_hparams/apply_caa.yaml",
    "sta": "hparams/Steer/sta_hparams/apply_sta.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check steering vectors using training prompts."
    )
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["caa", "sta"],
        choices=["caa", "sta"],
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[20],
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[1.0],
    )
    parser.add_argument("--model_name_or_path", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vector_root", default="vectors")
    parser.add_argument("--output_root", default="generation")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--sta_trim", type=float, default=0.65)
    parser.add_argument(
        "--sta_mode",
        default="act_and_freq",
        choices=["act_and_freq", "only_act", "only_freq"],
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--pair_limit",
        type=int,
        default=0,
        help="Limit number of contrastive pairs evaluated (0 = all).",
    )
    parser.add_argument(
        "--axbench_subdir",
        default="",
        help="AxBench subdirectory (e.g., '9b/l20') if the dataset is organised by model layer.",
    )
    return parser.parse_args()


def _build_vector_dir(vector_root: str, model_name: str, dataset_name: str, method: str) -> str:
    return os.path.join(
        vector_root,
        sanitize_model_name(model_name),
        dataset_name,
        f"{method}_vector",
    )


def _build_cfg(
    args: argparse.Namespace,
    methods: List[str],
    multiplier: float,
    vector_dirs: List[str],
    output_dir: str,
    dataset_keys: List[str],
) -> OmegaConf:
    cfg_dict: Dict = {
        "model_name_or_path": args.model_name_or_path,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "seed": args.seed,
        "use_chat_template": args.use_chat_template,
        "system_prompt": args.system_prompt,
        "apply_steer_hparam_paths": [METHOD_CONFIGS[m] for m in methods],
        "steer_vector_load_dir": vector_dirs,
        "layers": args.layers,
        "multipliers": [multiplier],
        "generation_data": dataset_keys,
        "generation_data_size": None,
        "generation_output_dir": output_dir,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "num_responses": args.num_responses,
        "steer_from_end_position": False,
        "generate_orig_output": True,
        "vllm_enable": False,
    }
    if "sta" in methods:
        cfg_dict["trims"] = [args.sta_trim]
        cfg_dict["mode"] = args.sta_mode
    return OmegaConf.create(cfg_dict)


def main() -> None:
    args = parse_args()

    print("Loading AxBench Concept500 train split ...")
    train_records = load_axbench_split("train", token=args.hf_token, subdir=args.axbench_subdir)
    pairs, metadata = build_contrastive_pairs(train_records, args.concept_id)

    if args.pair_limit > 0:
        pairs = pairs[: args.pair_limit]
    eval_sets = build_contrastive_eval_sets(pairs)

    dataset_name = f"axbench_concept_{args.concept_id}"
    vector_base = os.path.join(
        args.vector_root,
        sanitize_model_name(args.model_name_or_path),
        dataset_name,
    )
    if not os.path.exists(vector_base):
        raise FileNotFoundError(
            f"Vector directory not found: {vector_base}. Generate vectors first."
        )

    target_dir = os.path.join(
        args.output_root,
        sanitize_model_name(args.model_name_or_path),
        dataset_name,
        "sanity",
    )
    ensure_directory(target_dir)

    dataset_keys = list(eval_sets.keys())
    print(
        f"Evaluating {len(pairs)} contrastive prompts across {len(dataset_keys)} datasets."
    )

    for methods in [args.methods]:
        for multiplier in args.multipliers:
            run_name = f"{'_'.join(methods)}_mult{multiplier:g}"
            output_dir = os.path.join(target_dir, run_name)
            ensure_directory(output_dir)

            vector_dirs = [
                _build_vector_dir(args.vector_root, args.model_name_or_path, dataset_name, m)
                for m in methods
            ]
            for method, path in zip(methods, vector_dirs):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Expected vector directory for method '{method}' not found: {path}."
                    )

            cfg = _build_cfg(
                args,
                methods,
                multiplier,
                vector_dirs,
                output_dir,
                dataset_keys,
            )
            applier = BaseVectorApplier(cfg)
            applier.apply_vectors()
            applier.generate(eval_sets, save_results=True)
            if hasattr(applier.model, "reset_all"):
                applier.model.reset_all()

    print("Sanity check complete. Review outputs under:")
    print(target_dir)


if __name__ == "__main__":
    main()

