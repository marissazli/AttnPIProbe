import os
import json
from src.utils import load_json

def load_method_config(attr_type):
    """
    Load configuration for a specific attribution method.
    
    Args:
        attr_type (str): The attribution method type
        
    Returns:
        dict: Configuration dictionary for the method
    """
    config_path = f'method_configs/{attr_type}_config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return load_json(config_path)

def load_method_configs(args):
    """
    Load method-specific configurations and apply them to args.
    Command line arguments override config file values.
    """
    try:
        config = load_method_config(args.attr_type)
        
        # Apply attention trace configs
        if hasattr(args, 'avg_k') and args.avg_k is None and 'avg_k' in config:
            args.avg_k = config['avg_k']
        if hasattr(args, 'q') and args.q is None and 'q' in config:
            args.q = config['q']
        if hasattr(args, 'B') and args.B is None and 'B' in config:
            args.B = config['B']
            
        # Apply perturbation-based configs
        if hasattr(args, 'score_funcs') and args.score_funcs is None and 'score_funcs' in config:
            args.score_funcs = config['score_funcs']
        if hasattr(args, 'sh_N') and args.sh_N is None and 'sh_N' in config:
            args.sh_N = config['sh_N']
        if hasattr(args, 'beta') and args.beta is None and 'beta' in config:
            args.beta = config['beta']
        if hasattr(args, 'w') and args.w is None and 'w' in config:
            args.w = config['w']
            
        # Apply self-citation configs
        if hasattr(args, 'self_citation_model') and args.self_citation_model is None and 'self_citation_model' in config:
            args.self_citation_model = config['self_citation_model']
            
    except FileNotFoundError:
        print(f"Warning: No config file found for {args.attr_type}, using default values")
        # Set default values if config file doesn't exist
        if hasattr(args, 'avg_k') and args.avg_k is None:
            args.avg_k = 1
        if hasattr(args, 'q') and args.q is None:
            args.q = 0.5
        if hasattr(args, 'B') and args.B is None:
            args.B = 20
        if hasattr(args, 'score_funcs') and args.score_funcs is None:
            args.score_funcs = ["stc", 'loo', 'denoised_shapley']
        if hasattr(args, 'sh_N') and args.sh_N is None:
            args.sh_N = 20
        if hasattr(args, 'beta') and args.beta is None:
            args.beta = 0.2
        if hasattr(args, 'w') and args.w is None:
            args.w = 2
        if hasattr(args, 'self_citation_model') and args.self_citation_model is None:
            args.self_citation_model = "self"
    
    return args