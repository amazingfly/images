import json
import psutil
import os
import torch
import gc
import logging
import glob
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from upscale_images import get_upscale_config, upscale_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_STYLE_PROMPT = (
    "cyberpunk anime, cel shading, bold outlines, vibrant colors, cinematic long shot"
)

DEFAULT_NEGATIVE_PROMPT = (
    "extreme close-up, close-up portrait, tight crop, cropped composition, out of frame, "
    "truncated subject, oversized face, macro shot"
)

REPO_ROOT = Path(__file__).resolve().parents[1]

def resolve_config_path(config_path):
    path = Path(config_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path

def resolve_config_file_paths(config, config_path):
    base_dir = config_path.parent
    for key in ('output_dir', 'upscaled_dir', 'prompts_file', 'gemma_scripts_path'):
        value = config.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            config[key] = str((base_dir / path).resolve())

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_generation_config(config):
    generation_config = config.get('generation', {})
    width = int(generation_config.get('width', 512))
    height = int(generation_config.get('height', 768))

    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("generation.width and generation.height must be divisible by 8")

    return {
        'width': width,
        'height': height,
        'num_inference_steps': int(generation_config.get('num_inference_steps', 50)),
        'guidance_scale': float(generation_config.get('guidance_scale', 7.5)),
        'style_prompt': generation_config.get('style_prompt', DEFAULT_STYLE_PROMPT),
        'negative_prompt': generation_config.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT),
        'skip_existing': bool(generation_config.get('skip_existing', True)),
        'local_files_only': bool(config.get('local_files_only', False)),
    }

def build_prompt(prompt, style_prompt):
    if isinstance(prompt, dict):
        prompt = prompt.get('prompt') or prompt.get('text') or json.dumps(prompt)
    else:
        prompt = str(prompt)
    prompt = prompt.strip()
    style_prompt = style_prompt.strip()
    if not style_prompt:
        return prompt
    return f"{style_prompt}, {prompt}"

def existing_images_for_index(output_dir, index):
    exact_path = os.path.join(output_dir, f"image_{index}.png")
    timestamped_paths = glob.glob(os.path.join(output_dir, f"image_{index}_*.png"))
    paths = []
    if os.path.exists(exact_path):
        paths.append(exact_path)
    paths.extend(timestamped_paths)
    return paths

def generate_images(config_path):
    config_path = resolve_config_path(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    resolve_config_file_paths(config, config_path)
    generation_config = get_generation_config(config)
    upscale_config = get_upscale_config(config)
    
    with open(config['prompts_file'], 'r') as f:
        prompts = json.load(f)
    prompt_count = int(config.get('prompt_count', len(prompts)))
    target_prompts = prompts[:prompt_count]
    if prompt_count > len(prompts):
        logger.warning("Requested %s images but prompts file only has %s prompts.", prompt_count, len(prompts))
        
    logger.info(f"Initial RAM: {get_ram_usage():.2f} MB")
    logger.info(f"local_files_only={generation_config['local_files_only']}")
    
    # Load SD 1.5 components sequentially
    model_id = "runwayml/stable-diffusion-v1-5"
    logger.info("Loading SD components sequentially...")
    
    logger.info("Loading SD tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        local_files_only=generation_config['local_files_only'],
    )
    logger.info(f"SD tokenizer loaded. RAM: {get_ram_usage():.2f} MB")

    logger.info("Loading SD text encoder weights...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        local_files_only=generation_config['local_files_only'],
    )
    logger.info(f"SD text encoder loaded. RAM: {get_ram_usage():.2f} MB")

    logger.info("Loading SD UNet weights...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float32,
        local_files_only=generation_config['local_files_only'],
    )
    logger.info(f"SD UNet loaded. RAM: {get_ram_usage():.2f} MB")

    logger.info("Loading SD VAE weights...")
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=generation_config['local_files_only'],
    )
    logger.info(f"SD VAE loaded. RAM: {get_ram_usage():.2f} MB")

    logger.info("Loading SD scheduler...")
    scheduler = PNDMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        local_files_only=generation_config['local_files_only'],
    )
    logger.info(f"SD scheduler loaded. RAM: {get_ram_usage():.2f} MB")
    
    logger.info("Creating StableDiffusionPipeline...")
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
    logger.info(
        "Generation settings: %sx%s, %s steps, guidance_scale=%s",
        generation_config['width'],
        generation_config['height'],
        generation_config['num_inference_steps'],
        generation_config['guidance_scale'],
    )
    
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['upscaled_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pending_upscales = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-upscale") as upscale_executor:
        for i, prompt in enumerate(target_prompts):
            image_index = i + 1
            files = existing_images_for_index(config['output_dir'], image_index)
            if generation_config['skip_existing'] and files:
                logger.info(f"Image {i+1} already generated. Skipping.")
                for input_path in files:
                    output_path = os.path.join(config['upscaled_dir'], os.path.basename(input_path))
                    pending_upscales.append(
                        upscale_executor.submit(upscale_file, input_path, output_path, upscale_config)
                    )
                continue

            final_prompt = build_prompt(prompt, generation_config['style_prompt'])
            logger.info(f"Generating image {i+1}/{len(target_prompts)}: {final_prompt[:80]}...")

            image = pipe(
                final_prompt,
                negative_prompt=generation_config['negative_prompt'],
                num_inference_steps=generation_config['num_inference_steps'],
                guidance_scale=generation_config['guidance_scale'],
                height=generation_config['height'],
                width=generation_config['width'],
            ).images[0]

            output_filename = f"image_{i+1}_{timestamp}.png"
            input_path = os.path.join(config['output_dir'], output_filename)
            output_path = os.path.join(config['upscaled_dir'], output_filename)
            image.save(input_path)
            pending_upscales.append(
                upscale_executor.submit(upscale_file, input_path, output_path, upscale_config)
            )

            logger.info(
                "Finished image %s and queued its background upscale. RAM usage: %.2f MB",
                image_index,
                get_ram_usage(),
            )

            del image
            gc.collect()

        logger.info("Image inference complete; waiting for %s background upscale task(s).", len(pending_upscales))
        for future in pending_upscales:
            future.result()
        logger.info("All background upscale tasks complete.")
        
    # Unload
    del pipe
    del scheduler
    del vae
    del unet
    del text_encoder
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"RAM after unload: {get_ram_usage():.2f} MB")

if __name__ == "__main__":
    generate_images(sys.argv[1] if len(sys.argv) > 1 else "config.json")
