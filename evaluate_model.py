#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# Import model configurations
sys.path.append("src")
from llm_eval_configs import MODEL_OPTIONS

def load_data(file_path, chain_nums=None):
    """Load data from a JSON file and filter by chain_nums if provided."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # If file is test_long.json, filter by chain_nums
    if chain_nums is not None and "test_long.json" in file_path:
        filtered_data = []
        for item in data:
            if item.get("chain_nums", 0) == chain_nums:
                filtered_data.append(item)
        return filtered_data
    
    # For test_short.json format with "1" key
    if isinstance(data, dict) and "1" in data:
        return data["1"]
    
    return data

def evaluate_model(model_key, args):
    """Evaluate a specific model."""
    # Get model config
    model_config = MODEL_OPTIONS[model_key]
    base_model_name = model_config["name"]
    model_dir = model_config["dirname"]
    trust_remote_code = model_config.get("trust_remote_code", False)
    
    # Construct finetuned model path
    finetuned_model_path = os.path.join(model_dir, "finetuned_chain_nums2")
    
    print(f"\n\n{'='*50}")
    print(f"Evaluating model: {model_key} ({base_model_name})")
    print(f"Fine-tuned model path: {finetuned_model_path}")
    
    if not os.path.exists(finetuned_model_path):
        print(f"Error: Fine-tuned model not found at {finetuned_model_path}")
        return False
    
    # Load the fine-tuned model and tokenizer
    try:
        if args.use_peft:
            # Load the base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
            )
            # Load the PEFT adapter
            model = PeftModel.from_pretrained(model, finetuned_model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, 
                trust_remote_code=trust_remote_code
            )
        else:
            # Load the full fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(
                finetuned_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                finetuned_model_path,
                trust_remote_code=trust_remote_code
            )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    test_data = load_data(args.test_file, args.chain_nums)
    print(f"Loaded {len(test_data)} test examples with chain_nums={args.chain_nums}")
    
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(test_data):
        question = item["question"]
        true_answer = item["answer"].strip()
        
        # Format the question with instruction
        instruction = "Please answer the question with the most appropriate response."
        prompt = f"[INST] {instruction}\n\n{question} [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate prediction
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode the prediction
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the prediction
            prediction = prediction.replace(prompt, "").strip()
            
            # Check if the prediction matches the true answer
            # In a real scenario, you might want a more flexible comparison
            is_correct = true_answer.strip() in prediction
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": question,
                "true_answer": true_answer,
                "prediction": prediction,
                "is_correct": is_correct,
            })
        except Exception as e:
            print(f"Error generating prediction: {str(e)}")
            results.append({
                "question": question,
                "true_answer": true_answer,
                "prediction": "ERROR",
                "is_correct": False,
            })
            total += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results to file
    output_file = os.path.join(args.output_dir, f"eval_results_{model_key}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "model": model_key,
            "base_model": base_model_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Free up memory
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return True

def main(args):
    # Check if a specific model was requested
    if args.model_key:
        if args.model_key not in MODEL_OPTIONS:
            print(f"Error: Model key {args.model_key} not found in MODEL_OPTIONS")
            return
        
        # Evaluate the specified model
        evaluate_model(args.model_key, args)
    else:
        # Evaluate all models
        success_count = 0
        for model_key in MODEL_OPTIONS:
            try:
                success = evaluate_model(model_key, args)
                if success:
                    success_count += 1
            except Exception as e:
                print(f"Error evaluating model {model_key}: {str(e)}")
                continue
        
        print(f"\nSuccessfully evaluated {success_count}/{len(MODEL_OPTIONS)} models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on test data")
    parser.add_argument("--model_key", type=str, default="",
                        help="Key for the model in MODEL_OPTIONS to evaluate (leave empty to evaluate all)")
    parser.add_argument("--test_file", type=str, default="qwen2.5/test_long.json",
                        help="Test data JSON file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--chain_nums", type=int, default=2,
                        help="The chain_nums value to filter test data on")
    parser.add_argument("--use_peft", action="store_true",
                        help="Whether to load the model using PEFT adapters")
    
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES to use first GPU for evaluation
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    main(args) 
