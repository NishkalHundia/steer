"""
Generate CAA steering vectors for AxBench Concept500 dataset.

Usage:
    python generate_caa_axbench.py --concept_id 0 --layer 20 --model_name google/gemma-2-9b
    
The script will:
1. Load AxBench Concept500 dataset from HuggingFace
2. Find positive examples for the specified concept_id
3. Find negative examples with same genre where input prompts match exactly
4. Generate CAA steering vectors
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
    
    # Load train split
    train_data = load_dataset(hf_path, split="train")
    print(f"Train split: {len(train_data)} examples")
    
    # Convert to list of dicts
    train_list = [dict(item) for item in train_data]
    
    # Load test split only if needed (has different schema)
    test_list = []
    if load_test:
        try:
            test_data = load_dataset(hf_path, split="test")
            print(f"Test split: {len(test_data)} examples")
            test_list = [dict(item) for item in test_data]
        except Exception as e:
            print(f"Warning: Could not load test split due to schema differences: {e}")
            print("Test split has different columns (sae_link, sae_id) - skipping for now")
    
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
    
    # Get genre of this concept (assuming all examples for same concept have same genre)
    concept_genre = positive_examples[0].get('concept_genre')
    if concept_genre is None:
        raise ValueError(f"concept_genre not found for concept_id {concept_id}")
    
    print(f"Found {len(positive_examples)} positive examples for concept_id {concept_id}")
    print(f"Concept genre: {concept_genre}")
    
    # Create a mapping of input prompts to negative examples with same genre
    # Negative examples: different concept_id but same genre
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
        
        # Find matching negative example with same prompt
        if input_prompt in negative_examples_by_prompt:
            neg_item = negative_examples_by_prompt[input_prompt]
            
            # Format according to CAA requirements
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


def create_config(model_name, layer, output_dir, concept_id):
    """Create Hydra config for CAA vector generation."""
    config_dict = {
        'model_name_or_path': model_name,
        'torch_dtype': 'bfloat16',
        'device': 'cuda:0',
        'seed': 42,
        'use_chat_template': False,
        'system_prompt': '',
        'steer_train_hparam_paths': ['hparams/Steer/caa_hparams/generate_caa.yaml'],
        'steer_train_dataset': f'axbench_concept_{concept_id}',
        'save_vectors': True,
        'steer_vector_output_dirs': [output_dir],
        'vllm_enable': False
    }
    
    # Create CAA config
    caa_config_path = 'hparams/Steer/caa_hparams/generate_caa.yaml'
    os.makedirs(os.path.dirname(caa_config_path), exist_ok=True)
    
    caa_config = {
        'alg_name': 'caa',
        'layers': [layer],
        'multiple_choice': False,
        'save_activations': True,
        'save_vectors': True
    }
    
    # Save CAA config
    OmegaConf.save(OmegaConf.create(caa_config), caa_config_path)
    
    return OmegaConf.create(config_dict)


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
    
    # Update CAA config
    caa_config_path = 'hparams/Steer/caa_hparams/generate_caa.yaml'
    caa_config = OmegaConf.load(caa_config_path)
    caa_config.layers = [args.layer]
    OmegaConf.save(caa_config, caa_config_path)
    
    # Create dataset dict
    train_datasets = {
        f'axbench_concept_{args.concept_id}': contrastive_pairs
    }
    
    print(f"\nGenerating CAA vectors for concept {args.concept_id}...")
    print(f"Model: {args.model_name}")
    print(f"Layer: {args.layer}")
    print(f"Output dir: {args.output_dir}")
    print(f"Contrastive pairs: {len(contrastive_pairs)}")
    
    # Generate vectors
    vector_generator = BaseVectorGenerator(top_cfg)
    vectors = vector_generator.generate_vectors(train_datasets)
    
    print(f"\nSuccessfully generated CAA vectors!")
    print(f"Vectors saved to: {args.output_dir}")


if __name__ == '__main__':
    # Parse arguments before Hydra initialization
    parser = argparse.ArgumentParser(description='Generate CAA steering vectors for AxBench Concept500')
    parser.add_argument('--concept_id', type=int, required=True, help='Concept ID to generate vectors for')
    parser.add_argument('--layer', type=int, required=True, help='Layer number to extract activations from')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b-it', help='Model name or path')
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
