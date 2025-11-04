import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files

from .generate_sta_hparam import STAHyperParams
from typing import List

from ..sae_feature.sae_utils import (
    clear_gpu_cache,
    load_sae_from_dir,
    load_gemma_2_sae,
)
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def download_sae_from_hf(hf_path: str, local_cache_dir: str = None) -> str:
    """
    Download SAE from HuggingFace Hub - Downloads only the specific layer directory needed.
    Supports canonical SAE paths which are automatically resolved to actual paths.
    
    Args:
        hf_path: HuggingFace path in format "repo_id:path_within_repo" or "repo_id-canonical:layer_X/width_Y/canonical"
                 Examples:
                 - "google/gemma-scope-9b-pt-res:layer_20/width_131k/average_l0_114" (direct path)
                 - "google/gemma-scope-9b-pt-res-canonical:layer_20/width_16k/canonical" (canonical - auto-resolved)
        local_cache_dir: Optional local directory to cache downloads
    
    Returns:
        Local path to downloaded SAE directory
    """
    # Parse HuggingFace path
    if ':' in hf_path:
        repo_id, subpath = hf_path.split(':', 1)
    elif '/' in hf_path:
        parts = hf_path.split('/', 1)
        repo_id = parts[0]
        subpath = parts[1] if len(parts) > 1 else ''
    else:
        repo_id = hf_path
        subpath = ''
    
    if not subpath:
        raise ValueError(f"SAE subpath must be specified in format 'repo_id:subpath'. Got: {hf_path}")
    
    # Check if this is a canonical SAE path and resolve it
    if subpath.endswith('/canonical') or '/canonical' in subpath:
        print(f"Detected canonical SAE path: {subpath}")
        try:
            from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
            
            # Convert repo_id to release name (remove google/ prefix if present, keep canonical suffix)
            release_name = repo_id.replace('google/', '')
            if release_name not in ['gemma-scope-9b-pt-res-canonical', 'gemma-scope-9b-it-res-canonical']:
                # Try adding -canonical if not present
                if '-canonical' not in release_name:
                    release_name = release_name + '-canonical'
            
            sae_directory = get_pretrained_saes_directory()
            
            if release_name in sae_directory:
                sae_info = sae_directory[release_name]
                if subpath in sae_info.saes_map:
                    # Resolve canonical path to actual path
                    actual_path = sae_info.saes_map[subpath]
                    actual_repo_id = sae_info.repo_id
                    print(f"Resolved canonical path '{subpath}' to actual path '{actual_path}'")
                    subpath = actual_path
                    repo_id = actual_repo_id
                else:
                    raise ValueError(f"Canonical SAE ID '{subpath}' not found in release '{release_name}'. Available IDs: {list(sae_info.saes_map.keys())[:10]}...")
            else:
                print(f"Warning: Release '{release_name}' not found in SAE directory. Trying direct download...")
        except Exception as e:
            print(f"Warning: Could not resolve canonical path: {e}. Trying direct download...")
    
    # Extract layer directory name from subpath (e.g., "layer_20" from "layer_20/width_16k/average_l0_68")
    layer_dir = subpath.split('/')[0] if '/' in subpath else subpath
    
    # Use local cache or temp directory
    if local_cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "saes")
    else:
        cache_dir = local_cache_dir
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download ONLY the specific layer directory from HuggingFace
    print(f"Downloading SAE from HuggingFace: {repo_id}/{subpath}")
    print(f"Only downloading layer directory: {layer_dir}/*")
    
    try:
        # First, list files in the repo to find files in our target subpath
        print(f"Scanning repository for files in {subpath}...")
        all_files = list_repo_files(repo_id=repo_id, token=HUGGINGFACE_TOKEN)
        
        # Filter files to only those in our subpath
        target_files = [f for f in all_files if f.startswith(subpath + "/") or f == subpath]
        
        if not target_files:
            print(f"Warning: No files found matching {subpath}. Checking repository structure...")
            # Try to find similar paths
            layer_files = [f for f in all_files if layer_dir in f]
            if layer_files:
                print(f"Found {len(layer_files)} files with '{layer_dir}' in path.")
                # Group by width type to show available options
                width_types = set()
                for f in layer_files:
                    parts = f.split('/')
                    if len(parts) >= 3:
                        width_types.add(parts[1])  # width_16k, width_131k, etc.
                
                if width_types:
                    print(f"\nAvailable width types for {layer_dir}:")
                    for width_type in sorted(width_types):
                        print(f"  - {width_type}")
                    
                    # Show a few example paths
                    print(f"\nExample available paths:")
                    seen_paths = set()
                    for f in layer_files[:10]:
                        # Extract path up to average_l0_*
                        parts = f.split('/')
                        if len(parts) >= 3:
                            example_path = '/'.join(parts[:3])
                            if example_path not in seen_paths:
                                print(f"  - {repo_id}:{example_path}")
                                seen_paths.add(example_path)
                                if len(seen_paths) >= 5:
                                    break
                
                print(f"\nFirst few files found:")
                for f in layer_files[:5]:
                    print(f"  - {f}")
            raise ValueError(
                f"No files found in subpath {subpath} in repo {repo_id}. "
                f"Please check the repository structure and use a valid path."
            )
        
        print(f"Found {len(target_files)} files to download")
        
        # Download each file individually to the target directory
        target_dir = os.path.join(cache_dir, repo_id.replace("/", "_"), subpath)
        os.makedirs(target_dir, exist_ok=True)
        
        for file_path in target_files:
            # Download file
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir=os.path.join(cache_dir, repo_id.replace("/", "_")),
                local_dir_use_symlinks=False,
                token=HUGGINGFACE_TOKEN,
            )
            print(f"Downloaded: {file_path}")
        
        # Return the specific subdirectory path
        sae_path = os.path.join(cache_dir, repo_id.replace("/", "_"), subpath)
        if not os.path.exists(sae_path):
            raise ValueError(f"SAE subpath {subpath} not found after download in repo {repo_id}")
        
        print(f"Successfully downloaded SAE to: {sae_path}")
        return sae_path
    except ImportError:
        raise ImportError("huggingface_hub is required for downloading SAEs. Install with: pip install huggingface_hub")
    except Exception as e:
        raise ValueError(f"Failed to download SAE from HuggingFace {hf_path}: {e}")


def prepare_input(tokenizer, prompts, device="cuda"):
    input_tokens = tokenizer.tokenize(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens
def signed_min_max_normalize(tensor):
    abs_tensor = tensor.abs()
    min_val = abs_tensor.min()
    max_val = abs_tensor.max()
    normalized = (abs_tensor - min_val) / (max_val - min_val)
    return tensor.sign() * normalized  

def act_and_fre(act_data,
                pos_data,
                neg_data,
                mode,
                trim,
                sae,
                ):

    pec = 1-trim
    
    act_data_init = act_data.to(sae.W_dec.device)
    diff_data = pos_data - neg_data

    norm_act = signed_min_max_normalize(act_data) 
    norm_diff = signed_min_max_normalize(diff_data)  

    
    mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
    print("mask:",mask.sum())

   
    scores = torch.zeros_like(norm_diff)  
    scores[mask] = (norm_diff[mask]) 

    threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(pec * len(scores))]
    print(f'frequency threshold: {threshold_fre}')
    freq_mask = torch.abs(scores) >= threshold_fre
    print("freq_mask:",freq_mask.sum())

    threshold = torch.sort(torch.abs(act_data_init), descending=True, stable=True).values[int(pec * len(act_data_init))]
    print(f'threshold: {threshold}')
    act_top_mask = torch.abs(act_data_init) >= threshold
    print("act_top_mask:",act_top_mask.sum())

    result = None
    ######### act and fre ########
    if mode == "act_and_freq":
        freq_mask = freq_mask.to(sae.W_dec.device)
        act_data_combined = act_data.clone()
        combined_mask = freq_mask & act_top_mask
        print("combined_mask:",combined_mask.sum())
        act_data_combined[~combined_mask] = 0
        print(torch.abs(act_data_combined).sum())

        act_data_combined = act_data_combined.to(sae.W_dec.device)
        result = act_data_combined @ sae.W_dec
        print("result_combined.shape",result.shape)
        print("result_combined:",result)
        print("torch.norm(result_combined)", torch.norm(result))

    elif mode == "only_act":
    ########### only act ########
        act_data_act = act_data.clone()
        act_data_act[~act_top_mask] = 0
        act_data_act = act_data_act.to(sae.W_dec.device)
        result = act_data_act @ sae.W_dec

    elif mode == "only_freq":
    ########### only fre ########
        act_data_fre = act_data.clone()
        act_data_fre[~freq_mask] = 0
        act_data_fre = act_data_fre.to(sae.W_dec.device)
        result = act_data_fre @ sae.W_dec
    
    return result

def generate_sta_vectors(hparams:STAHyperParams, dataset, model = None, dataset_name = None):
    from ...models.get_model import get_model
    from ...datasets.caa_data import get_tokens_for_caa
    
    args = hparams
    del_model = True
    assert len(args.layers) == 1, "Not support many layers!!!"
    assert len(args.layers) == len(args.sae_paths), f"len(sae_paths) does not match the len(layers)"
    
    # Note: Layer validation will be done after downloading SAEs if needed
    # The old assertion that required layer number in path string is removed for HuggingFace compatibility
    
    if model is None:
        model, tokenizer = get_model(hparams)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = hparams
        
    device = model.device
    saes = dict([(layer, []) for layer in args.layers])
    for i,layer in enumerate(args.layers):
        sae_path = args.sae_paths[i]
        
        # Check if it's a HuggingFace path (contains : or starts with org/model format)
        is_hf_path = ':' in sae_path or ('/' in sae_path and not os.path.exists(sae_path) and not sae_path.startswith('.'))
        
        if is_hf_path:
            print(f"Detected HuggingFace SAE path: {sae_path}")
            # Download from HuggingFace
            sae_path = download_sae_from_hf(sae_path)
        
        if not os.path.exists(sae_path):
            raise ValueError(f"SAE path does not exist: {sae_path}")
        
        if "gemma" in args.model_name_or_path.lower():
            saes[layer], _, _ = load_gemma_2_sae(sae_path=sae_path, device=device)
        else:
            saes[layer], _, _ = load_sae_from_dir(sae_path, device=device)
    
    need_train_layers = []
    vectors = {}
    for layer in args.layers:
        feature_act_path = os.path.join(args.steer_vector_output_dir, f"feature_score/layer_{layer}_feature_act.pt")
        pos_freq_path = os.path.join(args.steer_vector_output_dir, f"feature_score/layer_{layer}_pos_freq.pt")
        neg_freq_path = os.path.join(args.steer_vector_output_dir, f"feature_score/layer_{layer}_neg_freq.pt")
        caa_vector_path = os.path.join(args.steer_vector_output_dir, f"caa_vector/layer_{layer}.pt" if args.multiple_choice == False else f"caa_vector_multiple_choice/layer_{layer}.pt")
        if os.path.exists(feature_act_path) and os.path.exists(pos_freq_path) and os.path.exists(neg_freq_path) and os.path.exists(caa_vector_path):
            feature_act = torch.load(feature_act_path, map_location=device)
            pos_feature_freq = torch.load(pos_freq_path, map_location=device)
            neg_feature_freq = torch.load(neg_freq_path, map_location=device)
            caa_vector = torch.load(caa_vector_path, map_location=device)
            
            if args.save_vectors:
                output_dir_sta = os.path.join(
                    args.steer_vector_output_dir, "sta_vector" if args.multiple_choice == False else "sta_vector_multiple_choice"
                )
                if not os.path.exists(output_dir_sta):
                    os.makedirs(output_dir_sta)

            for trim in args.trims:
                sta_vec = act_and_fre(feature_act,
                                    pos_feature_freq,
                                    neg_feature_freq,
                                    args.mode,
                                    trim,
                                    saes[layer],
                                    )
                caa_vector = caa_vector.to(device)
                sta_vec = sta_vec.to(device)
                multiplier = torch.norm(caa_vector, p=2) / torch.norm(sta_vec, p=2)
                print(f"caa_norm:{caa_vector.norm()}  sta_norm:{(multiplier * sta_vec).norm()}")
                if args.save_vectors:
                    torch.save(
                        multiplier * sta_vec,
                        os.path.join(output_dir_sta, f"layer_{layer}_{args.mode}_trim{trim}.pt"),
                    )
                vectors[f"layer_{layer}_{args.mode}_trim{trim}"] = multiplier * sta_vec
        else:
            need_train_layers.append(layer)
    if not need_train_layers:
        print(f"Computed vector using historical saved data, historical path is {os.path.join(args.steer_vector_output_dir, 'feature_score')}")
        return vectors

    args.layers = need_train_layers

    pos_activations = dict([(layer, []) for layer in args.layers])
    neg_activations = dict([(layer, []) for layer in args.layers])
    
    pos_sae_activations = dict([(layer, []) for layer in args.layers])
    neg_sae_activations = dict([(layer, []) for layer in args.layers])

    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

    for p_tokens_dict, n_tokens_dict in tqdm(
        zip(pos_tokens_list, neg_tokens_list),
        total=len(pos_tokens_list),
        desc="Processing prompts",
    ):
        p_tokens = p_tokens_dict["pos_tokens"]
        n_tokens = n_tokens_dict["neg_tokens"]
        ques_tokens_len = p_tokens_dict["ques_tokens_len"]
        model.reset_all()
        model.get_logits(p_tokens)

        for layer in args.layers:
            p_activations = model.get_last_activations(layer)
            p_sae_activations = saes[layer].encode(p_activations)

            # mean the activation over all answer tokens
            if args.multiple_choice == True:
                p_activations = p_activations[0, -2, :].detach().cpu()
                p_sae_activations = p_sae_activations[0, -2, :].detach().cpu()
            else:
                p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
                p_sae_activations = p_sae_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            
            pos_activations[layer].append(p_activations)
            pos_sae_activations[layer].append(p_sae_activations)

        model.reset_all()
        model.get_logits(n_tokens)

        for layer in args.layers:
            n_activations = model.get_last_activations(layer)
            n_sae_activations = saes[layer].encode(n_activations)

            if args.multiple_choice == True:
                n_activations = n_activations[0, -2, :].detach().cpu()
                n_sae_activations = n_sae_activations[0, -2, :].detach().cpu()
            else:
                n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
                n_sae_activations = n_sae_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            neg_activations[layer].append(n_activations)
            neg_sae_activations[layer].append(n_sae_activations)
        
    if args.save_vectors:
        if not os.path.exists(args.steer_vector_output_dir):
            os.makedirs(args.steer_vector_output_dir)

    for layer in args.layers:
        all_pos_layer = torch.stack(pos_activations[layer])
        all_neg_layer = torch.stack(neg_activations[layer])
        caa_vector = (all_pos_layer - all_neg_layer).mean(dim=0)

        if args.save_vectors:
            output_dir_caa = os.path.join(
                args.steer_vector_output_dir, "caa_vector" if args.multiple_choice == False else "caa_vector_multiple_choice"
            )
            if not os.path.exists(output_dir_caa):
                os.makedirs(output_dir_caa)
            torch.save(
                caa_vector,
                os.path.join(output_dir_caa, f"layer_{layer}.pt"),
            )
        vectors[f"layer_{layer}"] = caa_vector

        all_pos_sae_layer = torch.stack(pos_sae_activations[layer])
        all_neg_sae_layer = torch.stack(neg_sae_activations[layer])

        feature_act = (all_pos_sae_layer - all_neg_sae_layer).mean(dim=0)
        pos_feature_freq = (all_pos_sae_layer > 0).float().sum(0)
        neg_feature_freq = (all_neg_sae_layer > 0).float().sum(0)

        # feature_act = feature_act.to(device)
        # steering_vector = feature_act @ saes[layer].W_dec
        if args.save_vectors:
            if not os.path.exists(os.path.join(args.steer_vector_output_dir, f"feature_score")):
                os.makedirs(os.path.join(args.steer_vector_output_dir, f"feature_score"))
            
            torch.save(
                pos_feature_freq,
                os.path.join(args.steer_vector_output_dir, f"feature_score/layer_{layer}_pos_freq.pt"),
            )
            torch.save(
                neg_feature_freq,
                os.path.join(args.steer_vector_output_dir, f"feature_score/layer_{layer}_neg_freq.pt"),
            )
            torch.save(
                feature_act,
                os.path.join(args.steer_vector_output_dir, f"feature_score/layer_{layer}_feature_act.pt"),
            )

        if args.save_vectors:
            output_dir_sta = os.path.join(
                args.steer_vector_output_dir, "sta_vector" if args.multiple_choice == False else "sta_vector_multiple_choice"
            )
            if not os.path.exists(output_dir_sta):
                os.makedirs(output_dir_sta)
                
        try:
            for trim in args.trims:
                sta_vec = act_and_fre(feature_act,
                                    pos_feature_freq,
                                    neg_feature_freq,
                                    args.mode,
                                    trim,
                                    saes[layer],
                                    )
                caa_vector = caa_vector.to(device)
                sta_vec = sta_vec.to(device)
                multiplier = torch.norm(caa_vector, p=2) / torch.norm(sta_vec, p=2)
                print(f"caa_norm:{caa_vector.norm()}  sta_norm:{(multiplier * sta_vec).norm()}")
                if args.save_vectors is True:
                    torch.save(
                        multiplier * sta_vec,
                        os.path.join(output_dir_sta, f"layer_{layer}_{args.mode}_trim{trim}.pt"),
                    )
                vectors[f"layer_{layer}_{args.mode}_trim{trim}"] = multiplier * sta_vec
        finally:
            if del_model:
                model.model.to('cpu')
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=[24])
    parser.add_argument("--sae_paths", nargs="+", type=str, default=['/data2/xzwnlp/gemma-scope-9b-pt-res/layer_24/width_16k/average_l0_114'])
    parser.add_argument("--trims", nargs="+", type=float, default=[0.65])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="act_and_freq")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="safeedit",
    )
    parser.add_argument("--model_name_or_path", type=str, default="/mnt/20t/msy/models/gemma-2-9b-base")
    parser.add_argument("--multiple_choice", action="store_true", default=False)
    parser.add_argument("--steer_vector_output_dir", type=str, default="../")
    
    args = parser.parse_args()

    if args.hparams_path:
        hparams = STAHyperParams.from_hparams(args.hparams_path)
    else:
        hparams = STAHyperParams(
            model_name_or_path=args.model_name_or_path,
            layers=args.layers,
            sae_paths=args.sae_paths,
            device=args.device,
            dataset_name=args.steer_train_dataset,
            trims=args.trims,
            mode=args.mode,
            multiple_choice=args.multiple_choice,
            steer_vector_output_dir=args.steer_vector_output_dir,
        )

    generate_sta_vectors(hparams)
