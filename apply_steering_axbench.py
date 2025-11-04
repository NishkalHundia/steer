"""
Apply CAA/STA steering vectors to AxBench Concept500 test dataset.

Usage:
    python apply_steering_axbench.py --concept_id 0 --method caa --layer 20 --multipliers 1.0 1.5 2.0 --model_name google/gemma-2-9b --vector_dir vectors/axbench_concept_0/caa
    
    python apply_steering_axbench.py --concept_id 0 --method sta --layer 20 --multipliers 1.0 1.5 2.0 --model_name google/gemma-2-9b --vector_dir vectors/axbench_concept_0/sta --trim 0.65 --mode act_and_freq
    
    python apply_steering_axbench.py --concept_id 0 --method both --layer 20 --multipliers 1.0 1.5 2.0 --model_name google/gemma-2-9b --caa_vector_dir vectors/axbench_concept_0/caa --sta_vector_dir vectors/axbench_concept_0/sta --trim 0.65 --mode act_and_freq
"""

import argparse
import os
import json
from datasets import load_dataset
from steer.vector_appliers.vector_applier import BaseVectorApplier
from omegaconf import DictConfig, OmegaConf
import hydra
import torch


def load_axbench_data(hf_path="pyvene/axbench-concept500", load_test=False):
    """Load AxBench dataset from HuggingFace."""
    print(f"Loading AxBench dataset from {hf_path}...")
    
    # Load train split - use parquet files directly to avoid schema conflicts
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    import pandas as pd
    
    # Find train parquet file (it's in a subdirectory like 9b/l20/train/data.parquet)
    print("Finding train parquet file...")
    all_files = list_repo_files(repo_id=hf_path, repo_type="dataset")
    train_files = [f for f in all_files if "/train/" in f.lower() and f.endswith(".parquet")]
    
    if not train_files:
        raise ValueError(f"Could not find train parquet file in {hf_path}")
    
    # Use the first train file found
    train_filename = train_files[0]
    print(f"Loading train split from: {train_filename}")
    train_file = hf_hub_download(repo_id=hf_path, filename=train_filename, repo_type="dataset")
    train_table = pq.read_table(train_file)
    # Convert Arrow table to pandas DataFrame, then to list of dicts
    train_df = train_table.to_pandas()
    train_list = train_df.to_dict('records')
    print(f"Train split: {len(train_list)} examples")
    
    # Load test split only if needed (has different schema)
    test_list = []
    if load_test:
        try:
            # Find test parquet file
            test_files = [f for f in all_files if "/test/" in f.lower() and f.endswith(".parquet")]
            if not test_files:
                raise ValueError(f"Could not find test parquet file in {hf_path}")
            
            test_filename = test_files[0]
            print(f"Loading test split from: {test_filename}")
            test_file = hf_hub_download(repo_id=hf_path, filename=test_filename, repo_type="dataset")
            test_table = pq.read_table(test_file)
            # Convert Arrow table to pandas DataFrame, then to list of dicts
            test_df = test_table.to_pandas()
            test_list = test_df.to_dict('records')
            print(f"Test split: {len(test_list)} examples")
        except Exception as e:
            print(f"Warning: Could not load test split: {e}")
            print("Test split has different columns (sae_link, sae_id)")
    
    return train_list, test_list


def get_test_prompts(test_data, concept_id, limit=None):
    """Extract test prompts for a specific concept_id."""
    test_examples = [
        item for item in test_data 
        if item.get('concept_id') == concept_id
    ]
    
    if not test_examples:
        raise ValueError(f"No test examples found for concept_id {concept_id}")
    
    print(f"Found {len(test_examples)} test examples for concept_id {concept_id}")
    
    # Get concept description from first example
    concept_description = test_examples[0].get('output_concept', 'N/A')
    print(f"Concept being steered: {concept_description}")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        test_examples = test_examples[:limit]
        print(f"Limited to {len(test_examples)} test examples for faster iteration")
    
    # Format for generation
    generation_data = []
    for item in test_examples:
        # Test split might have different schema - handle both
        input_prompt = item.get('input', item.get('instruction', ''))
        expected_response = item.get('winning_output', item.get('reference_response', ''))
        
        generation_data.append({
            'input': input_prompt,
            'expected_response': expected_response
        })
    
    return generation_data, concept_description


def main_func(top_cfg: DictConfig, args):
    # Set vector directories (vectors are saved to {base_dir}/{dataset_name}/{method}_vector/)
    dataset_name = f'axbench_concept_{args.concept_id}'
    if args.method == 'caa':
        if args.vector_dir is None:
            args.vector_dir = f'vectors/{dataset_name}/{dataset_name}/caa_vector'
        args.caa_vector_dir = args.vector_dir
    elif args.method == 'sta':
        if args.vector_dir is None:
            args.vector_dir = f'vectors/{dataset_name}/{dataset_name}/sta_vector'
        args.sta_vector_dir = args.vector_dir
    elif args.method == 'both':
        if args.caa_vector_dir is None:
            args.caa_vector_dir = f'vectors/{dataset_name}/{dataset_name}/caa_vector'
        if args.sta_vector_dir is None:
            args.sta_vector_dir = f'vectors/{dataset_name}/{dataset_name}/sta_vector'
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f'generation/axbench_concept_{args.concept_id}/{args.method}'
    
    # Load dataset (need test for generation)
    train_data, test_data = load_axbench_data(args.hf_path, load_test=True)
    
    # Get test prompts
    generation_data, concept_description = get_test_prompts(test_data, args.concept_id, limit=args.limit)
    
    print(f"\n{'='*60}")
    print(f"Concept Description: {concept_description}")
    print(f"{'='*60}\n")
    
    # Update config
    top_cfg.model_name_or_path = args.model_name
    top_cfg.use_chat_template = True  # Use chat template for instruction-tuned models
    top_cfg.generation_output_dir = args.output_dir
    top_cfg.generation_data = [f'axbench_concept_{args.concept_id}']
    
    # Prepare generation dataset
    generation_datasets = {
        f'axbench_concept_{args.concept_id}': [{'input': item['input']} for item in generation_data]
    }
    
    # Store expected responses for later
    expected_responses = {item['input']: item['expected_response'] for item in generation_data}
    
    all_results = []
    
    # Apply vectors and generate for each multiplier
    for multiplier in args.multipliers:
        print(f"\n{'='*60}")
        print(f"Applying steering with multiplier: {multiplier}")
        print(f"{'='*60}")
        
        # Update config based on method
        if args.method == 'caa':
            top_cfg.apply_steer_hparam_paths = ['hparams/Steer/caa_hparams/apply_caa.yaml']
            top_cfg.steer_vector_load_dir = [args.caa_vector_dir]
            
            # Update CAA apply config
            caa_apply_config = OmegaConf.load('hparams/Steer/caa_hparams/apply_caa.yaml')
            caa_apply_config.layers = [args.layer]
            caa_apply_config.multipliers = [multiplier]
            OmegaConf.save(caa_apply_config, 'hparams/Steer/caa_hparams/apply_caa.yaml')
            
        elif args.method == 'sta':
            top_cfg.apply_steer_hparam_paths = ['hparams/Steer/sta_hparams/apply_sta.yaml']
            top_cfg.steer_vector_load_dir = [args.sta_vector_dir]
            
            # Update STA apply config
            sta_apply_config = OmegaConf.load('hparams/Steer/sta_hparams/apply_sta.yaml')
            sta_apply_config.layers = [args.layer]
            sta_apply_config.multipliers = [multiplier]
            sta_apply_config.trims = [args.trim]
            sta_apply_config.mode = args.mode
            OmegaConf.save(sta_apply_config, 'hparams/Steer/sta_hparams/apply_sta.yaml')
            
        elif args.method == 'both':
            top_cfg.apply_steer_hparam_paths = [
                'hparams/Steer/caa_hparams/apply_caa.yaml',
                'hparams/Steer/sta_hparams/apply_sta.yaml'
            ]
            top_cfg.steer_vector_load_dir = [
                args.caa_vector_dir,
                args.sta_vector_dir
            ]
            
            # Update both configs
            caa_apply_config = OmegaConf.load('hparams/Steer/caa_hparams/apply_caa.yaml')
            caa_apply_config.layers = [args.layer]
            caa_apply_config.multipliers = [multiplier]
            OmegaConf.save(caa_apply_config, 'hparams/Steer/caa_hparams/apply_caa.yaml')
            
            sta_apply_config = OmegaConf.load('hparams/Steer/sta_hparams/apply_sta.yaml')
            sta_apply_config.layers = [args.layer]
            sta_apply_config.multipliers = [multiplier]
            sta_apply_config.trims = [args.trim]
            sta_apply_config.mode = args.mode
            OmegaConf.save(sta_apply_config, 'hparams/Steer/sta_hparams/apply_sta.yaml')
        
        # Create vector applier
        vector_applier = BaseVectorApplier(top_cfg)
        vector_applier.apply_vectors()
        
        # Generate
        results = vector_applier.generate(generation_datasets)
        
        # Add multiplier and expected responses to results
        for result in results:
            result['multiplier'] = multiplier
            result['method'] = args.method
            result['concept_description'] = concept_description
            input_key = result.get('input', '')
            result['expected_response'] = expected_responses.get(input_key, '')
            # pred is a list, get first element if available
            if isinstance(result.get('pred'), list) and len(result.get('pred', [])) > 0:
                result['generated_response'] = result['pred'][0]
            else:
                result['generated_response'] = result.get('pred', '')
        
        all_results.extend(results)
        
        # Reset model
        vector_applier.model.reset_all()
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'concept_{args.concept_id}_results.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total generations: {len(all_results)}")
    print(f"Multipliers tested: {args.multipliers}")


if __name__ == '__main__':
    # Parse arguments before Hydra initialization
    parser = argparse.ArgumentParser(description='Apply steering vectors to AxBench Concept500 test dataset')
    parser.add_argument('--concept_id', type=int, required=True, help='Concept ID')
    parser.add_argument('--method', type=str, required=True, choices=['caa', 'sta', 'both'], help='Steering method')
    parser.add_argument('--layer', type=int, required=True, help='Layer number')
    parser.add_argument('--multipliers', type=float, nargs='+', required=True, help='Steering multipliers to test')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b-it', help='Model name or path')
    parser.add_argument('--caa_vector_dir', type=str, default=None, help='CAA vector directory')
    parser.add_argument('--sta_vector_dir', type=str, default=None, help='STA vector directory')
    parser.add_argument('--vector_dir', type=str, default=None, help='Vector directory (for single method)')
    parser.add_argument('--trim', type=float, default=0.65, help='Trim parameter for STA (required if using STA)')
    parser.add_argument('--mode', type=str, default='act_and_freq', choices=['act_and_freq', 'only_act', 'only_freq'], help='STA mode')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for generations')
    parser.add_argument('--hf_path', type=str, default='pyvene/axbench-concept500', help='HuggingFace dataset path')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of test prompts (for faster iteration)')
    
    args = parser.parse_args()
    
    # Use Hydra's compose API to load config
    from hydra import compose, initialize_config_dir
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), 'hparams', 'Steer')
    with initialize_config_dir(config_dir=config_path, version_base='1.2'):
        cfg = compose(config_name='vector_apply.yaml')
        main_func(cfg, args)
