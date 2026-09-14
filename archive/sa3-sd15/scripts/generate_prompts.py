import json
import psutil
import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def generate_prompts(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Initial RAM: {get_ram_usage():.2f} MB")
    
    model_id = "google/gemma-2-2b-it" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    print(f"RAM after model load: {get_ram_usage():.2f} MB")
    
    preface = "Generate a numbered list of 50 unique image generation prompts for an anime style cyborg female with futuristic weapons in a cyberpunk dystopia. Add colorful and elaborate description details. Keep each prompt within 100 words."
    
    inputs = tokenizer(preface, return_tensors="pt")
    
    # Generate - Increase max_new_tokens for 50 prompts
    outputs = model.generate(**inputs, max_new_tokens=2000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse and save to JSON
    # Improved parsing: split by numbered list items
    import re
    prompts = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', response, re.DOTALL)
    prompts = [p.strip() for p in prompts if len(p.strip()) > 20]
    
    # Ensure we only have up to 50
    final_prompts = prompts[:config['prompt_count']]
    
    with open(config['prompts_file'], 'w') as f:
        json.dump(final_prompts, f, indent=4)
        
    print(f"Generated {len(final_prompts)} prompts. RAM: {get_ram_usage():.2f} MB")
    
    # Explicitly unload
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"RAM after unload: {get_ram_usage():.2f} MB")

if __name__ == "__main__":
    generate_prompts("config.json")
