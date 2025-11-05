"""Apply CAA/STA vectors generated from AxBench Concept500 data."""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from typing import Dict, Iterable, List

from omegaconf import OmegaConf

from axbench_utils import (
    build_test_dataset,
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
        description="Apply CAA/STA steering vectors on AxBench Concept500 prompts."
    )
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["caa", "sta"],
        choices=["caa", "sta"],
        help="Methods to evaluate. Each method is executed individually.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also evaluate the combination of all specified methods.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[20],
        help="Layers used during generation (applied identically).",
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[1.0],
        help="Steering multipliers to compare.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="google/gemma-2-9b-it",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vector_root",
        default="vectors",
        help="Directory that contains generated vectors.",
    )
    parser.add_argument(
        "--output_root",
        default="generation",
        help="Directory where generation results will be stored.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument(
        "--generate_orig_output",
        action="store_true",
        help="Also generate baseline outputs without steering.",
    )
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
        "--generation_limit",
        type=int,
        default=0,
        help="Optional limit on number of evaluation samples (0 = all).",
    )
    parser.add_argument(
        "--axbench_subdir",
        default="",
        help="AxBench subdirectory (e.g., '9b/l20') if the dataset is organised by model layer.",
    )
    parser.add_argument(
        "--include_negative",
        action="store_true",
        help="Evaluate negative targets as a separate dataset as well.",
    )
    return parser.parse_args()


def _method_combinations(methods: List[str], combine: bool) -> List[List[str]]:
    combos = [[m] for m in methods]
    if combine and len(methods) > 1:
        combos.append(methods)
    return combos


def _build_vector_dir(vector_root: str, model_name: str, dataset_name: str, method: str) -> str:
    return os.path.join(
        vector_root,
        sanitize_model_name(model_name),
        dataset_name,
        f"{method}_vector",
    )


def _build_top_cfg(
    args: argparse.Namespace,
    method_list: List[str],
    multiplier: float,
    vector_dirs: List[str],
    output_dir: str,
    dataset_key: str,
) -> OmegaConf:
    cfg_dict: Dict = {
        "model_name_or_path": args.model_name_or_path,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "seed": args.seed,
        "use_chat_template": args.use_chat_template,
        "system_prompt": args.system_prompt,
        "apply_steer_hparam_paths": [METHOD_CONFIGS[m] for m in method_list],
        "steer_vector_load_dir": vector_dirs,
        "layers": args.layers,
        "multipliers": [multiplier],
        "generation_data": [dataset_key],
        "generation_data_size": None,
        "generation_output_dir": output_dir,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "num_responses": args.num_responses,
        "steer_from_end_position": False,
        "generate_orig_output": args.generate_orig_output,
        "vllm_enable": False,
    }
    if args.generation_limit > 0:
        cfg_dict["generation_data_size"] = args.generation_limit
    if "sta" in method_list:
        cfg_dict["trims"] = [args.sta_trim]
        cfg_dict["mode"] = args.sta_mode
    return OmegaConf.create(cfg_dict)


def main() -> None:
    args = parse_args()

    print("Loading AxBench Concept500 test split ...")
    test_records = load_axbench_split("test", token=args.hf_token, subdir=args.axbench_subdir)
    evaluation_data = build_test_dataset(test_records, args.concept_id)
    dataset_key = f"axbench_concept_{args.concept_id}"

    positive_items = [item for item in evaluation_data if item.get("target_type", "positive") != "negative"]
    negative_items = [item for item in evaluation_data if item.get("target_type") == "negative"]

    eval_sets: Dict[str, List[Dict]] = {dataset_key: positive_items}
    if args.include_negative and negative_items:
        eval_sets[f"{dataset_key}_negative"] = negative_items

    vector_base = os.path.join(
        args.vector_root,
        sanitize_model_name(args.model_name_or_path),
        dataset_key,
    )
    if not os.path.exists(vector_base):
        raise FileNotFoundError(
            f"Vector directory not found: {vector_base}. Generate vectors first."
        )

    print(
        "Evaluating on {} positive prompts{}.".format(
            len(positive_items),
            f" and {len(negative_items)} negative prompts" if args.include_negative and negative_items else "",
        )
    )
    method_sets = _method_combinations(args.methods, args.combine)

    for method_list in method_sets:
        for multiplier in args.multipliers:
            run_name = f"{'_'.join(method_list)}_mult{multiplier:g}"
            output_dir = os.path.join(
                args.output_root,
                sanitize_model_name(args.model_name_or_path),
                dataset_key,
                run_name,
            )
            ensure_directory(output_dir)

            vector_dirs = [
                _build_vector_dir(args.vector_root, args.model_name_or_path, dataset_key, m)
                for m in method_list
            ]
            for method, path in zip(method_list, vector_dirs):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Expected vector directory for method '{method}' not found: {path}."
                    )

            cfg = _build_top_cfg(
                args,
                method_list,
                multiplier,
                vector_dirs,
                output_dir,
                dataset_key,
            )

            print(
                f"\n=== Applying {' + '.join(method_list)} with multiplier {multiplier} ==="
            )
            applier = BaseVectorApplier(cfg)
            applier.apply_vectors()
            applier.generate(eval_sets)
            if hasattr(applier.model, "reset_all"):
                applier.model.reset_all()

    print("\nGeneration complete. Results written to:")
    print(
        os.path.join(
            args.output_root,
            sanitize_model_name(args.model_name_or_path),
            dataset_key,
        )
    )


if __name__ == "__main__":
    main()

