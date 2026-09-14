import json
import os
import cv2
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_upscale_config(config):
    upscale_config = config.get('upscale', {})
    width = int(upscale_config.get('width', 1920))
    height = int(upscale_config.get('height', 1080))
    mode = upscale_config.get('mode', 'contain_blur')
    foreground_scale = float(upscale_config.get('foreground_scale', 0.94))
    background_blur = int(upscale_config.get('background_blur', 45))

    if width <= 0 or height <= 0:
        raise ValueError("upscale.width and upscale.height must be positive")
    if not 0 < foreground_scale <= 1:
        raise ValueError("upscale.foreground_scale must be greater than 0 and at most 1")
    if background_blur < 1:
        raise ValueError("upscale.background_blur must be positive")
    if background_blur % 2 == 0:
        background_blur += 1

    return {
        'width': width,
        'height': height,
        'mode': mode,
        'foreground_scale': foreground_scale,
        'background_blur': background_blur,
        'skip_existing': bool(upscale_config.get('skip_existing', True)),
    }

def resized_dimensions(source_width, source_height, max_width, max_height, cover=False):
    if cover:
        scale = max(max_width / source_width, max_height / source_height)
    else:
        scale = min(max_width / source_width, max_height / source_height)
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return width, height

def resize_to_cover(img, target_width, target_height):
    source_height, source_width = img.shape[:2]
    width, height = resized_dimensions(source_width, source_height, target_width, target_height, cover=True)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    x = max(0, (width - target_width) // 2)
    y = max(0, (height - target_height) // 2)
    return resized[y:y + target_height, x:x + target_width]

def compose_contain_blur(img, target_width, target_height, foreground_scale, background_blur):
    source_height, source_width = img.shape[:2]
    background = resize_to_cover(img, target_width, target_height)
    background = cv2.GaussianBlur(background, (background_blur, background_blur), 0)

    max_foreground_width = int(round(target_width * foreground_scale))
    max_foreground_height = int(round(target_height * foreground_scale))
    foreground_width, foreground_height = resized_dimensions(
        source_width,
        source_height,
        max_foreground_width,
        max_foreground_height,
    )
    foreground = cv2.resize(img, (foreground_width, foreground_height), interpolation=cv2.INTER_LANCZOS4)

    output = background.copy()
    x = (target_width - foreground_width) // 2
    y = (target_height - foreground_height) // 2
    output[y:y + foreground_height, x:x + foreground_width] = foreground
    return output

def upscale_image(img, upscale_config):
    target_width = upscale_config['width']
    target_height = upscale_config['height']
    mode = upscale_config['mode']

    if mode == 'stretch':
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    if mode == 'contain_blur':
        return compose_contain_blur(
            img,
            target_width,
            target_height,
            upscale_config['foreground_scale'],
            upscale_config['background_blur'],
        )

    raise ValueError(f"Unsupported upscale.mode: {mode}")

def upscale_file(input_path, output_path, upscale_config):
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)

    if upscale_config['skip_existing'] and os.path.exists(output_path):
        logger.info(f"Image {os.path.basename(output_path)} already upscaled. Skipping.")
        return output_path

    img = cv2.imread(input_path)
    if img is None:
        raise OSError(f"Could not read {input_path}")

    source_height, source_width = img.shape[:2]
    filename = os.path.basename(input_path)
    logger.info(
        "Composing %s from %sx%s to %sx%s using %s mode.",
        filename,
        source_width,
        source_height,
        upscale_config['width'],
        upscale_config['height'],
        upscale_config['mode'],
    )
    upscaled_img = upscale_image(img, upscale_config)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, upscaled_img):
        raise OSError(f"Failed to write {output_path}")

    logger.info(f"Finished upscaling {filename}.")
    return output_path

def upscale_images(config_path):
    config_path = resolve_config_path(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    resolve_config_file_paths(config, config_path)
    upscale_config = get_upscale_config(config)
        
    os.makedirs(config['upscaled_dir'], exist_ok=True)
    
    # Process images in output_dir
    files = sorted([f for f in os.listdir(config['output_dir']) if f.endswith('.png')])
    
    for filename in files:
        input_path = os.path.join(config['output_dir'], filename)
        output_path = os.path.join(config['upscaled_dir'], filename)

        try:
            upscale_file(input_path, output_path, upscale_config)
        except OSError as exc:
            logger.warning("%s. Skipping.", exc)

if __name__ == "__main__":
    upscale_images(sys.argv[1] if len(sys.argv) > 1 else "config.json")
