import os
import torch
import json
from typing import List, Dict, Any
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import pdb
from src.llm_eval_configs import MODEL_OPTIONS
import argparse
from peft import PeftModel
# Constants
BATCH_SIZE = 200  # Adjust based on GPU memory
MAX_CHAIN_NUM = 5  # Maximum k-hop to evaluate (inclusive)
TENSOR_PARALLEL_SIZE = 4  # Number of GPUs to use for tensor parallelism

def serialize_hf_output(outputs, input_texts):
    """Convert Hugging Face generation outputs to JSON-serializable dictionaries."""
    serialized_outputs = []
    for i, output in enumerate(outputs):
        output_dict = {
            "request_id": str(i),
            "prompt": input_texts[i],
            "outputs": [{
                "text": output,
                "finish_reason": "length"  # Hugging Face doesn't provide finish reason
            }]
        }
        serialized_outputs.append(output_dict)
    return serialized_outputs

def split_batch(batch: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a batch into smaller batches of specified size."""
    return [batch[i:i + batch_size] for i in range(0, len(batch), batch_size)]

def process_batch(
    inputs: List[str],
    tracked_indices: List[List[str]],
    model,
    tokenizer,
    batch_size: int = 32,
    maxlength: int = 1
) -> List[Dict[str, Any]]:
    """Process inputs one by one and extract logits from last position."""
    all_outputs = []

    # Process each input individually
    for input_text in tqdm(inputs, desc="Processing inputs"):
        try:
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Get logits from last position
            last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

            # Convert to probabilities
            probs = F.softmax(last_token_logits, dim=-1)
            
            # Get top tokens and their probabilities
            top_probs, top_indices = torch.topk(probs, k=5)
            
            # Convert to dictionary
            output_dict = {
                "input_text": input_text,
                "logits": last_token_logits.cpu(),
                "top_probs": top_probs.cpu(),
                "top_indices": top_indices.cpu()
            }
            
            all_outputs.append(output_dict)
            
        except Exception as e:
            print(f"Error processing input: {e}")
            # Return partial results if available
            if all_outputs:
                return all_outputs
            raise
    
    return all_outputs

def get_tracked_prob(input_texts, logits, tokenizer):
    """Calculate the tracked probabilities for given input_texts and logits."""
    inputs = [pair['question'] for pair in input_texts]
    tracked_indices = [pair['query_names'] + pair['non_query_names'] for pair in input_texts]
    lengths = [len(tokenizer(input).input_ids) for input in inputs]
    
    check_indices = torch.LongTensor([
        [i, l-1, j] for i, l in enumerate(lengths) 
        for j in tracked_indices[i]
    ])
    
    probs = F.softmax(logits, dim=-1)
    tracked_prob = probs[check_indices[:, 0], check_indices[:, 2]]
    return tracked_prob

def split_topics(input_texts):
    """Split inputs into topics based on keywords."""
    keywords = ["locate", "grand", "family", "three"]
    display_keywords = {
        "locate": "geography",
        "grand": "relations",
        "family": "biology",
        "three": "arithmetic",
        "other": "other"
    }
    
    topic_dict = {keyword: [] for keyword in keywords}
    topic_dict["other"] = []
    topic_indices = {keyword: [] for keyword in keywords}
    topic_indices["other"] = []

    for idx, pair in enumerate(input_texts):
        question = pair["question"]
        found = False
        for keyword in keywords:
            if keyword in question:
                topic_dict[keyword].append(pair)
                topic_indices[keyword].append(idx)
                found = True
                break
        if not found:
            topic_dict["locate"].append(pair)
            topic_indices["locate"].append(idx)
    
    for topic, texts in topic_dict.items():
        if texts:  # Only yield non-empty topics
            yield display_keywords[topic], texts, topic_indices[topic]

def run_inference_for_model(model_config, test_data, use_lora=False):
    """Run inference for all k values for a specific model."""
    print(f"Running inference for {model_config['name']}")
    
    model_dir = model_config["dirname"]
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config.get("trust_remote_code", False)
        )
        # Set pad token for models that don't have one
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if use_lora:
            # Load base model first
            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=model_config.get("trust_remote_code", False)
            )
            # Load LoRA weights
            lora_path = os.path.join("finetuned", model_config["alias"]+"_chain2", "checkpoint-186")
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"Loaded LoRA weights from {lora_path}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=model_config.get("trust_remote_code", False)
            )
        print(f"Model {model_config['name']} loaded successfully")
    except Exception as e:
        print(f"Error initializing model {model_config['name']}: {e}")
        return False
    
    # Process each k value
    success = True
    for k, input_texts in test_data.items():
        if int(k) > MAX_CHAIN_NUM:
            continue
        
        print(f"Processing chain_nums={k}")
        
        try:
            # Extract inputs and tracked indices
            inputs = [pair["question"] for pair in input_texts]
            tracked_indices = [pair["query_names"] + pair["non_query_names"] for pair in input_texts]
            
            # Process the inputs
            outputs = process_batch(
                inputs=inputs,
                tracked_indices=tracked_indices,
                model=model,
                tokenizer=tokenizer,
                maxlength=1
            )
            
            # Save outputs with lora suffix if using LoRA
            suffix = "_lora" if use_lora else ""
            outputs_file = os.path.join(model_dir, f"outputs_chain_nums{k}{suffix}.pt")
            torch.save(outputs, outputs_file)
            
            # Save logits separately for probability calculation
            logits_file = os.path.join(model_dir, f"logits_chain_nums{k}{suffix}.pt")
            all_logits = torch.stack([output["logits"] for output in outputs])
            torch.save(all_logits, logits_file)
            
            print(f"Saved outputs and logits for chain_nums={k}")
        except Exception as e:
            print(f"Error during inference for chain_nums={k}: {e}")
            success = False
    
    return success

def calculate_probabilities(model_config, use_lora=False):
    """Calculate tracked probabilities for a specific model."""
    print(f"Calculating probabilities for {model_config['name']}")
    
    model_dir = model_config["dirname"]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    except Exception as e:
        print(f"Error loading tokenizer for {model_config['name']}: {e}")
        return None
    
    test_file = os.path.join(model_dir, "test_long.json")
    
    if not os.path.isfile(test_file):
        print(f"Test file not found: {test_file}")
        return None
    
    # Load test data
    try:
        with open(test_file, "r") as f:
            test_long = json.load(f)
    except Exception as e:
        print(f"Error loading test file: {e}")
        return None
    
    model_results = {}
    suffix = "_lora" if use_lora else ""
    
    for k in test_long.keys():
        if int(k) > MAX_CHAIN_NUM:
            continue
        
        logits_file = os.path.join(model_dir, f"logits_chain_nums{k}{suffix}.pt")
        if not os.path.isfile(logits_file):
            print(f"Logits file not found: {logits_file}")
            continue
        
        try:
            input_texts = test_long[k]
            logits = torch.load(logits_file)
            
            k_results = {}
            all_probs = []  # Store all probabilities for calculating overall statistics
            
            for topic, input_text_group, indices in split_topics(input_texts):
                if topic == 'other':
                    continue
                
                try:
                    tracked_prob = get_tracked_prob(input_text_group, logits[indices, ...], tokenizer)
                    reshaped_probs = tracked_prob.view(-1, 3*int(k))
                    
                    # Calculate mean
                    mean_probs = reshaped_probs.mean(dim=0)
                    
                    # Calculate standard error
                    std_error = reshaped_probs.std(dim=0) / torch.sqrt(torch.tensor(reshaped_probs.size(0), dtype=torch.float32))
                    
                    # Store results
                    k_results[topic] = {
                        'mean': mean_probs,
                        'std_error': std_error
                    }
                    
                    # Add to all_probs for overall statistics
                    all_probs.append(reshaped_probs)
                    
                    print(f"chain_nums={k}, topic={topic}:")
                    print(f"  Mean: {mean_probs}")
                    print(f"  Std Error: {std_error}")
                except Exception as e:
                    print(f"Error calculating probabilities for topic {topic}: {e}")
            
            # Calculate overall statistics across all topics
            if all_probs:
                try:
                    # Combine all probabilities
                    combined_probs = torch.cat(all_probs, dim=0)
                    
                    # Calculate overall mean
                    overall_mean = combined_probs.mean(dim=0)
                    
                    # Calculate overall standard error
                    overall_std_error = combined_probs.std(dim=0) / torch.sqrt(torch.tensor(combined_probs.size(0), dtype=torch.float32))
                    
                    # Store overall results
                    k_results['overall'] = {
                        'mean': overall_mean,
                        'std_error': overall_std_error
                    }
                    
                    print(f"chain_nums={k}, overall:")
                    print(f"  Mean: {overall_mean}")
                    print(f"  Std Error: {overall_std_error}")
                except Exception as e:
                    print(f"Error calculating overall probabilities: {e}")
            
            model_results[k] = k_results
        except Exception as e:
            print(f"Error processing logits for chain_nums={k}: {e}")
    
    return model_results

def evaluate_models(model_name, use_lora=False):
    for model_alias, model_config in MODEL_OPTIONS.items():
        if model_name is not None and model_name != model_alias:
            continue
        test_file = os.path.join(model_config["dirname"], "test_long.json")
        if os.path.isfile(test_file):
            try:
                with open(test_file, "r") as f:
                    test_data = json.load(f)
                print(f"Found test data in {test_file}")
            except Exception as e:
                print(f"Error loading test file {test_file}: {e}")
        print(f"\nEvaluating model: {model_alias}")
        
        # Ensure model directory exists
        model_dir = model_config["dirname"]
        os.makedirs(model_dir, exist_ok=True)
        
        # Validate model configuration
        if "name" not in model_config or not model_config["name"]:
            print(f"Invalid model configuration for {model_alias}: missing or empty 'name'")
            continue
        
        # Run inference for all k values with a single model instance
        success = run_inference_for_model(model_config, test_data, use_lora)
        if not success:
            print(f"Some inference steps failed for {model_alias}")
        
        # Calculate probabilities
        model_results = calculate_probabilities(model_config, use_lora)
        
        if model_results:
            # Save individual model results with lora suffix if using LoRA
            suffix = "_lora" if use_lora else ""
            result_file = os.path.join(model_dir, f"tracked_prob{suffix}.pt")
            torch.save(model_results, result_file)
            print(f"Saved results to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lora", action="store_true", help="Use LoRA weights for inference")
    args = parser.parse_args()
    # Set environment variables for GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
    
    # Run evaluation
    evaluate_models(args.model, args.lora)
