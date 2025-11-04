"""
Sanity check script: Apply steering vectors to the same prompts used for training.

Usage:
    python sanity_check_axbench.py --concept_id 0 --method caa --layer 20 --multiplier 1.0 --model_name google/gemma-2-9b --vector_dir vectors/axbench_concept_0/caa
    
    python sanity_check_axbench.py --concept_id 0 --method sta --layer 20 --multiplier 1.0 --model_name google/gemma-2-9b --vector_dir vectors/axbench_concept_0/sta --trim 0.65 --mode act_and_freq
"""

import argparse
import os
import json
from datasets import load_dataset
from steer.vector_appliers.vector_applier import BaseVectorApplier
from omegaconf import DictConfig, OmegaConf
import hydra


def load_axbench_data(hf_path="pyvene/axbench-concept500"):
    """Load AxBench dataset from HuggingFace."""
    print(f"Loading AxBench dataset from {hf_path}...")
    train_data = load_dataset(hf_path, split="train")
    test_data = load_dataset(hf_path, split="test")
    
    print(f"Train split: {len(train_data)} examples")
    print(f"Test split: {len(test_data)} examples")
    
    # Convert to list of dicts
    train_list = [dict(item) for item in train_data]
    test_list = [dict(item) for item in test_data]
    
    return train_list, test_list


def get_training_pairs(train_data, concept_id):
    """
    Get training pairs for sanity check.
    Returns the same prompts used for training with their expected matching/not_matching responses.
    """
    # Get positive examples for this concept_id
    positive_examples = [
        item for item in train_data 
        if item.get('concept_id') == concept_id
    ]
    
    if not positive_examples:
        raise ValueError(f"No positive examples found for concept_id {concept_id}")
    
    concept_genre = positive_examples[0].get('concept_genre')
    if concept_genre is None:
        raise ValueError(f"concept_genre not found for concept_id {concept_id}")
    
    # Create mapping of input prompts to negative examples with same genre
    negative_examples_by_prompt = {}
    for item in train_data:
        if (item.get('concept_id') != concept_id and 
            item.get('concept_genre') == concept_genre):
            input_prompt = item.get('input', '')
            if input_prompt and input_prompt not in negative_examples_by_prompt:
                negative_examples_by_prompt[input_prompt] = item
    
    # Create training pairs
    training_pairs = []
    for pos_item in positive_examples:
        input_prompt = pos_item.get('input', '')
        
        if input_prompt in negative_examples_by_prompt:
            neg_item = negative_examples_by_prompt[input_prompt]
            
            pair = {
                'input': input_prompt,
                'expected_matching': pos_item.get('winning_output', ''),
                'expected_not_matching': neg_item.get('winning_output', '')
            }
            training_pairs.append(pair)
    
    print(f"Found {len(training_pairs)} training pairs for sanity check")
    
    return training_pairs


@hydra.main(version_base='1.2', config_path='./hparams/Steer', config_name='vector_apply.yaml')
def main(top_cfg: DictConfig):
    parser = argparse.ArgumentParser(description='Sanity check: Apply steering vectors to training prompts')
    parser.add_argument('--concept_id', type=int, required=True, help='Concept ID')
    parser.add_argument('--method', type=str, required=True, choices=['caa', 'sta'], help='Steering method')
    parser.add_argument('--layer', type=int, required=True, help='Layer number')
    parser.add_argument('--multiplier', type=float, default=1.0, help='Steering multiplier')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b', help='Model name or path')
    parser.add_argument('--vector_dir', type=str, default=None, help='Vector directory')
    parser.add_argument('--trim', type=float, default=0.65, help='Trim parameter for STA')
    parser.add_argument('--mode', type=str, default='act_and_freq', choices=['act_and_freq', 'only_act', 'only_freq'], help='STA mode')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--hf_path', type=str, default='pyvene/axbench-concept500', help='HuggingFace dataset path')
    
    args = parser.parse_args()
    
    # Set vector directory (vectors are saved to {base_dir}/{dataset_name}/{method}_vector/)
    dataset_name = f'axbench_concept_{args.concept_id}'
    if args.vector_dir is None:
        if args.method == 'caa':
            args.vector_dir = f'vectors/{dataset_name}/{dataset_name}/caa_vector'
        else:
            args.vector_dir = f'vectors/{dataset_name}/{dataset_name}/sta_vector'
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f'sanity_check/axbench_concept_{args.concept_id}/{args.method}'
    
    # Load dataset
    train_data, test_data = load_axbench_data(args.hf_path)
    
    # Get training pairs
    training_pairs = get_training_pairs(train_data, args.concept_id)
    
    # Update config
    top_cfg.model_name_or_path = args.model_name
    top_cfg.generation_output_dir = args.output_dir
    top_cfg.generation_data = [f'axbench_concept_{args.concept_id}_sanity']
    
    # Prepare generation dataset (just the prompts)
    generation_datasets = {
        f'axbench_concept_{args.concept_id}_sanity': [{'input': pair['input']} for pair in training_pairs]
    }
    
    # Update config based on method
    if args.method == 'caa':
        top_cfg.apply_steer_hparam_paths = ['hparams/Steer/caa_hparams/apply_caa.yaml']
        top_cfg.steer_vector_load_dir = [args.vector_dir]
        
        # Update CAA apply config
        caa_apply_config = OmegaConf.load('hparams/Steer/caa_hparams/apply_caa.yaml')
        caa_apply_config.layers = [args.layer]
        caa_apply_config.multipliers = [args.multiplier]
        OmegaConf.save(caa_apply_config, 'hparams/Steer/caa_hparams/apply_caa.yaml')
        
    elif args.method == 'sta':
        top_cfg.apply_steer_hparam_paths = ['hparams/Steer/sta_hparams/apply_sta.yaml']
        top_cfg.steer_vector_load_dir = [args.vector_dir]
        
        # Update STA apply config
        sta_apply_config = OmegaConf.load('hparams/Steer/sta_hparams/apply_sta.yaml')
        sta_apply_config.layers = [args.layer]
        sta_apply_config.multipliers = [args.multiplier]
        sta_apply_config.trims = [args.trim]
        sta_apply_config.mode = args.mode
        OmegaConf.save(sta_apply_config, 'hparams/Steer/sta_hparams/apply_sta.yaml')
    
    print(f"\nSanity check for concept {args.concept_id}")
    print(f"Method: {args.method}")
    print(f"Multiplier: {args.multiplier}")
    print(f"Training pairs: {len(training_pairs)}")
    
    # Apply vectors
    vector_applier = BaseVectorApplier(top_cfg)
    vector_applier.apply_vectors()
    
    # Generate
    results = vector_applier.generate(generation_datasets)
    
    # Combine with expected responses
    sanity_results = []
    for i, result in enumerate(results):
        if i < len(training_pairs):
            # pred is a list, get first element if available
            if isinstance(result.get('pred'), list) and len(result.get('pred', [])) > 0:
                generated_response = result['pred'][0]
            else:
                generated_response = result.get('pred', '')
            
            sanity_results.append({
                'input': result.get('input', ''),
                'generated_response': generated_response,
                'expected_matching': training_pairs[i]['expected_matching'],
                'expected_not_matching': training_pairs[i]['expected_not_matching'],
                'multiplier': args.multiplier,
                'method': args.method
            })
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'concept_{args.concept_id}_sanity_check.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sanity_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSanity check results saved to: {output_file}")
    print(f"Total generations: {len(sanity_results)}")
    print(f"\nExpected behavior: Generated responses should match 'expected_matching' more than 'expected_not_matching'")


if __name__ == '__main__':
    main()
