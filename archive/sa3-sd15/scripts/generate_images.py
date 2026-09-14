import json
import psutil
import os
import torch
import gc
import logging
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def generate_images(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(config['prompts_file'], 'r') as f:
        prompts = json.load(f)
        
    logger.info(f"Initial RAM: {get_ram_usage():.2f} MB")
    
    # Load SD 1.5 components sequentially
    model_id = "runwayml/stable-diffusion-v1-5"
    logger.info("Loading SD components sequentially...")
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    pipe = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    pipe.to("cpu")
    
    logger.info(f"RAM after pipeline load: {get_ram_usage():.2f} MB")
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    for i, prompt in enumerate(prompts[:config['prompt_count']]):
        output_path = os.path.join(config['output_dir'], f"image_{i+1}.png")
        if os.path.exists(output_path):
            logger.info(f"Image {i+1} already exists. Skipping.")
            continue
            
        logger.info(f"Generating image {i+1}/{len(prompts[:config['prompt_count']])}: {prompt[:30]}...")
        
        # CPU generation - increased steps to 50
        image = pipe(prompt, num_inference_steps=50, height=512, width=512).images[0]
        
        # Save
        image.save(output_path)
        
        logger.info(f"Finished image {i+1}. RAM usage: {get_ram_usage():.2f} MB")
        
        # Explicitly clean up after each generation
        gc.collect()
        
    # Unload
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"RAM after unload: {get_ram_usage():.2f} MB")

if __name__ == "__main__":
    generate_images("config.json")
