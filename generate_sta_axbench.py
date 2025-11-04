"""
Generate STA steering vectors for AxBench Concept500 dataset.

Usage:
    python generate_sta_axbench.py --concept_id 0 --layer 20 --model_name google/gemma-2-9b --sae_path google/gemma-scope-9b-pt-res-canonical:layer_20/width_16k/canonical
"""

import argparse
import os
from datasets import load_dataset
from steer.vector_generators.vector_generators import BaseVectorGenerator
from omegaconf import DictConfig, OmegaConf
import hydra


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
    
    return train_list, test_list


def create_contrastive_pairs(train_data, concept_id):
    """
    Create contrastive pairs for a specific concept_id.
    
    For each positive example (with target concept_id):
    - Find negative examples with same genre
    - Match by exact input prompt
    """
    # Get positive examples for this concept_id
    positive_examples = [
        item for item in train_data 
        if item.get('concept_id') == concept_id
    ]
    
    if not positive_examples:
        raise ValueError(f"No positive examples found for concept_id {concept_id}")
    
    # Get genre of this concept
    concept_genre = positive_examples[0].get('concept_genre')
    if concept_genre is None:
        raise ValueError(f"concept_genre not found for concept_id {concept_id}")
    
    print(f"Found {len(positive_examples)} positive examples for concept_id {concept_id}")
    print(f"Concept genre: {concept_genre}")
    
    # Create a mapping of input prompts to negative examples with same genre
    negative_examples_by_prompt = {}
    for item in train_data:
        if (item.get('concept_id') != concept_id and 
            item.get('concept_genre') == concept_genre):
            input_prompt = item.get('input', '')
            if input_prompt and input_prompt not in negative_examples_by_prompt:
                negative_examples_by_prompt[input_prompt] = item
    
    print(f"Found {len(negative_examples_by_prompt)} unique negative prompts with same genre")
    
    # Create contrastive pairs
    contrastive_pairs = []
    for pos_item in positive_examples:
        input_prompt = pos_item.get('input', '')
        
        if input_prompt in negative_examples_by_prompt:
            neg_item = negative_examples_by_prompt[input_prompt]
            
            pair = {
                'question': input_prompt,
                'matching': pos_item.get('winning_output', ''),
                'not_matching': neg_item.get('winning_output', '')
            }
            contrastive_pairs.append(pair)
        else:
            print(f"Warning: No matching negative example found for prompt: {input_prompt[:100]}...")
    
    print(f"Created {len(contrastive_pairs)} contrastive pairs")
    
    if not contrastive_pairs:
        raise ValueError(f"No contrastive pairs created for concept_id {concept_id}")
    
    return contrastive_pairs


def main_func(top_cfg: DictConfig, args):
    
    # Set output directory (base directory - dataset_name will be added automatically)
    if args.output_dir is None:
        args.output_dir = f'vectors/axbench_concept_{args.concept_id}'
    
    # Load dataset (only train needed for vector generation)
    train_data, _ = load_axbench_data(args.hf_path, load_test=False)
    
    # Create contrastive pairs
    contrastive_pairs = create_contrastive_pairs(train_data, args.concept_id)
    
    # Update config
    top_cfg.model_name_or_path = args.model_name
    top_cfg.use_chat_template = True  # Use chat template for instruction-tuned models
    top_cfg.steer_train_dataset = f'axbench_concept_{args.concept_id}'
    top_cfg.steer_vector_output_dirs = [args.output_dir]
    top_cfg.steer_train_hparam_paths = ['hparams/Steer/sta_hparams/generate_sta.yaml']
    
    # Update STA config
    sta_config_path = 'hparams/Steer/sta_hparams/generate_sta.yaml'
    sta_config = OmegaConf.load(sta_config_path)
    sta_config.layers = [args.layer]
    sta_config.sae_paths = [args.sae_path]
    sta_config.trims = [args.trim]
    sta_config.mode = args.mode
    OmegaConf.save(sta_config, sta_config_path)
    
    # Create dataset dict
    train_datasets = {
        f'axbench_concept_{args.concept_id}': contrastive_pairs
    }
    
    print(f"\nGenerating STA vectors for concept {args.concept_id}...")
    print(f"Model: {args.model_name}")
    print(f"Layer: {args.layer}")
    print(f"SAE path: {args.sae_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Contrastive pairs: {len(contrastive_pairs)}")
    
    # Generate vectors
    vector_generator = BaseVectorGenerator(top_cfg)
    vectors = vector_generator.generate_vectors(train_datasets)
    
    print(f"\nSuccessfully generated STA vectors!")
    print(f"Vectors saved to: {args.output_dir}")


if __name__ == '__main__':
    # Parse arguments before Hydra initialization
    parser = argparse.ArgumentParser(description='Generate STA steering vectors for AxBench Concept500')
    parser.add_argument('--concept_id', type=int, required=True, help='Concept ID to generate vectors for')
    parser.add_argument('--layer', type=int, required=True, help='Layer number to extract activations from')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b-it', help='Model name or path')
    parser.add_argument('--sae_path', type=str, default='google/gemma-scope-9b-it-res-canonical:layer_20/width_16k/canonical', help='SAE path (default: canonical SAE for gemma-2-9b-it). For pretrained model use: google/gemma-scope-9b-pt-res-canonical:layer_20/width_16k/canonical')
    parser.add_argument('--trim', type=float, default=0.65, help='Trim parameter for STA')
    parser.add_argument('--mode', type=str, default='act_and_freq', choices=['act_and_freq', 'only_act', 'only_freq'], help='STA mode')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for vectors')
    parser.add_argument('--hf_path', type=str, default='pyvene/axbench-concept500', help='HuggingFace dataset path')
    
    args = parser.parse_args()
    
    # Use Hydra's compose API to load config
    from hydra import compose, initialize_config_dir
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), 'hparams', 'Steer')
    with initialize_config_dir(config_dir=config_path, version_base='1.2'):
        cfg = compose(config_name='vector_generate.yaml')
        main_func(cfg, args)
