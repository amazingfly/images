import json
import logging
import os
import sys
import time
from pathlib import Path

from generate_images_lora import generate_images
from generate_littlequeen_dataset_prompts import generate_dataset_prompts
from generate_prompts import resolve_config_file_paths, resolve_config_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    resolved = resolve_config_path(config_path)
    with open(resolved, 'r', encoding='utf-8') as f:
        config = json.load(f)
    resolve_config_file_paths(config, resolved)
    return resolved, config


def prompt_file_count(prompts_file):
    if not os.path.exists(prompts_file):
        return 0
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise ValueError(f"Prompts file must contain a JSON list: {prompts_file}")
    return len(prompts)


def should_generate_prompts(config):
    prompt_count = int(config.get('prompt_count', 50))
    regenerate = bool(config.get('pipeline', {}).get('regenerate_prompts', False))
    available = prompt_file_count(config['prompts_file'])
    return regenerate or available < prompt_count


def run_pipeline(config_path):
    resolved_config_path, config = load_config(config_path)
    pipeline_config = config.get('pipeline', {})

    logger.info("Little Queen dataset pipeline starting with %s", resolved_config_path)
    if should_generate_prompts(config):
        logger.info("Generating Little Queen dataset scene prompts...")
        generate_dataset_prompts(str(resolved_config_path))
        cooldown = int(pipeline_config.get('ram_cooldown_seconds', 15))
        if cooldown > 0:
            logger.info("Waiting %s seconds for RAM to settle before SD1.5 generation...", cooldown)
            time.sleep(cooldown)
    else:
        logger.info("Skipping prompt generation; prompts file already has enough records.")

    logger.info("Generating LoRA dataset images...")
    generate_images(str(resolved_config_path))
    logger.info("Little Queen dataset pipeline complete.")


if __name__ == "__main__":
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "config_lora_littlequeen_dataset.json")
