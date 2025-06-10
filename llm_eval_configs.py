"""
LLM Evaluation Configuration File
Contains model options and configurations for fine-tuning and evaluation.
"""

MODEL_OPTIONS = {
    # Llama Models
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "trust_remote_code": False,
        "dirname": "llama2-7b"
    },
    "llama3.1-8b": {
        "name": "meta-llama/Meta-Llama-3.1-8B",
        "trust_remote_code": False,
        "dirname": "llama3.1-8b"
    },
    "llama3.1-70b": {
        "name": "meta-llama/Meta-Llama-3.1-70B",
        "trust_remote_code": False,
        "dirname": "llama3.1-70b"
    },
    
    # OLMo Model
    "olmo": {
        "name": "allenai/OLMo-7B",
        "trust_remote_code": False,
        "dirname": "olmo"
    },
    
    # Qwen Model
    "qwen": {
        "name": "Qwen/Qwen2.5-7B",
        "trust_remote_code": True,
        "dirname": "qwen2.5"
    },
}

# Additional configuration options
DEFAULT_CONFIG = {
    "trust_remote_code": True,
    "torch_dtype": "float16",
    "device_map": "auto",
}

# Model categories for easy filtering
MODEL_CATEGORIES = {
    "small": ["llama3.1-8b", "olmo", "qwen2.5"],
    "medium": ["llama2-7b"],
    "large": ["llama3.1-70b"],
    "llama": ["llama2-7b", "llama3.1-8b", "llama3.1-70b"],
    "qwen": ["qwen2.5"],
    "olmo": ["olmo"],
}

def get_model_config(model_key):
    """Get model configuration by key."""
    if model_key not in MODEL_OPTIONS:
        raise ValueError(f"Model '{model_key}' not found in MODEL_OPTIONS. Available models: {list(MODEL_OPTIONS.keys())}")
    
    config = MODEL_OPTIONS[model_key].copy()
    
    # Apply default config if not specified
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
    
    return config

def list_models_by_category(category):
    """List models by category."""
    if category not in MODEL_CATEGORIES:
        raise ValueError(f"Category '{category}' not found. Available categories: {list(MODEL_CATEGORIES.keys())}")
    
    return MODEL_CATEGORIES[category]

def get_all_model_keys():
    """Get all available model keys."""
    return list(MODEL_OPTIONS.keys()) 
