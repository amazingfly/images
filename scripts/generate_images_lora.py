
# Load centralized workstation defaults; explicit environment/CLI values win.
import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
for _workspace_root in _WorkspacePath(__file__).resolve().parents:
    if (_workspace_root / "media_workspace").is_dir():
        _workspace_sys.path.insert(0, str(_workspace_root))
        break
from media_workspace.config import apply_environment as _apply_workspace
_apply_workspace()

import json
import psutil
import os
import torch
import gc
import logging
import glob
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    PNDMScheduler,
)
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer
from upscale_images import get_upscale_config, upscale_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_STYLE_PROMPT = (
    "cyberpunk anime, cel shading, bold outlines, vibrant colors, cinematic long shot"
)

DEFAULT_CHARACTER_PROMPT = ""

DEFAULT_NEGATIVE_PROMPT = (
    "extreme close-up, close-up portrait, tight crop, cropped composition, out of frame, "
    "truncated subject, oversized face, macro shot"
)

DEFAULT_LORA_PATH = "LORA/lora_littlequeen.safetensors"
DEFAULT_LORA_ADAPTER_NAME = "littlequeen"
DEFAULT_LORA_SCALE = 0.7

REPO_ROOT = Path(__file__).resolve().parents[1]

def resolve_config_path(config_path):
    path = Path(config_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path

def resolve_config_file_paths(config, config_path):
    base_dir = config_path.parent
    for key in ('output_dir', 'upscaled_dir', 'prompts_file', 'manifest_file', 'gemma_scripts_path'):
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

    seed = generation_config.get('seed')
    if seed is not None:
        seed = int(seed)
    return {
        'width': width,
        'height': height,
        'num_inference_steps': int(generation_config.get('num_inference_steps', 50)),
        'guidance_scale': float(generation_config.get('guidance_scale', 7.5)),
        'style_prompt': generation_config.get('style_prompt', DEFAULT_STYLE_PROMPT),
        'character_prompt': generation_config.get('character_prompt', DEFAULT_CHARACTER_PROMPT),
        'negative_prompt': generation_config.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT),
        'skip_existing': bool(generation_config.get('skip_existing', True)),
        'local_files_only': bool(config.get('local_files_only', False)),
        'seed': seed,
    }

def get_two_pass_config(config, generation_config, lora_config):
    two_pass_config = config.get('two_pass', {})
    enabled = bool(two_pass_config.get('enabled', False))
    strength = float(two_pass_config.get('strength', 0.55))
    if strength <= 0 or strength > 1:
        raise ValueError("two_pass.strength must be greater than 0 and at most 1")

    return {
        'enabled': enabled,
        'draft_source': two_pass_config.get('draft_source', 'generated'),
        'draft_lora_scale': float(two_pass_config.get('draft_lora_scale', 0.25)),
        'draft_num_inference_steps': int(two_pass_config.get('draft_num_inference_steps', 5)),
        'draft_guidance_scale': float(two_pass_config.get('draft_guidance_scale', generation_config['guidance_scale'])),
        'refine_lora_scale': float(two_pass_config.get('refine_lora_scale', lora_config['scale'])),
        'refine_num_inference_steps': int(two_pass_config.get('refine_num_inference_steps', 20)),
        'refine_guidance_scale': float(two_pass_config.get('refine_guidance_scale', generation_config['guidance_scale'])),
        'strength': strength,
        'save_drafts': bool(two_pass_config.get('save_drafts', True)),
    }

def get_lora_config(config, config_path):
    lora_config = config.get('lora', {})
    enabled = bool(lora_config.get('enabled', True))
    raw_path = lora_config.get('path', DEFAULT_LORA_PATH)
    path = Path(raw_path)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()

    scale = float(lora_config.get('scale', DEFAULT_LORA_SCALE))
    if scale < 0 or scale > 2:
        raise ValueError("lora.scale must be between 0 and 2")

    return {
        'enabled': enabled,
        'path': path,
        'adapter_name': lora_config.get('adapter_name', DEFAULT_LORA_ADAPTER_NAME),
        'scale': scale,
        'fuse': bool(lora_config.get('fuse', False)),
        'load_unet': bool(lora_config.get('load_unet', True)),
        'load_text_encoder': bool(lora_config.get('load_text_encoder', True)),
    }

def get_module(root, module_name):
    module = root
    for index, part in enumerate(module_name.split('.')):
        if index == 0 and part == 'text_model' and not hasattr(module, part):
            continue
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def kohya_text_encoder_module_name(lora_name):
    if not lora_name.startswith('lora_te_'):
        return None

    module_name = lora_name.removeprefix('lora_te_')
    module_name = module_name.replace('text_model_encoder_layers_', 'text_model.encoder.layers.')
    module_name = re.sub(r'(text_model\.encoder\.layers\.\d+)_', r'\1.', module_name, count=1)
    module_name = module_name.replace('.self_attn_', '.self_attn.')
    module_name = module_name.replace('.mlp_', '.mlp.')
    return module_name

def apply_kohya_text_encoder_lora(text_encoder, raw_state_dict, scale_delta):
    applied_count = 0
    skipped = []
    down_suffix = '.lora_down.weight'

    if scale_delta == 0:
        return 0

    for key in sorted(raw_state_dict):
        if not key.startswith('lora_te_') or not key.endswith(down_suffix):
            continue

        lora_name = key[:-len(down_suffix)]
        module_name = kohya_text_encoder_module_name(lora_name)
        up_key = f'{lora_name}.lora_up.weight'
        alpha_key = f'{lora_name}.alpha'
        if module_name is None or up_key not in raw_state_dict:
            skipped.append(lora_name)
            continue

        module = get_module(text_encoder, module_name)
        if not hasattr(module, 'weight') or module.weight.ndim != 2:
            skipped.append(lora_name)
            continue

        down = raw_state_dict[key].to(dtype=torch.float32)
        up = raw_state_dict[up_key].to(dtype=torch.float32)
        rank = down.shape[0]
        alpha_tensor = raw_state_dict.get(alpha_key)
        alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
        delta = torch.mm(up, down) * (scale_delta * alpha / rank)
        module.weight.data.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
        applied_count += 1

    if skipped:
        logger.warning("Skipped %s text-encoder LoRA tensor group(s): %s", len(skipped), skipped[:5])

    logger.info("Applied %s Kohya text-encoder LoRA tensor group(s).", applied_count)
    return applied_count

class LoraAdapterHandle:
    def __init__(self, adapter_name, raw_state_dict, load_unet, load_text_encoder, current_scale=0.0):
        self.adapter_name = adapter_name
        self.raw_state_dict = raw_state_dict
        self.load_unet = load_unet
        self.load_text_encoder = load_text_encoder
        self.current_text_encoder_scale = current_scale

    def set_scale(self, pipe, scale):
        if self.load_unet:
            pipe.set_adapters(self.adapter_name, adapter_weights=scale)

        if self.load_text_encoder:
            scale_delta = scale - self.current_text_encoder_scale
            apply_kohya_text_encoder_lora(pipe.text_encoder, self.raw_state_dict, scale_delta)
            self.current_text_encoder_scale = scale

        logger.info("LoRA scale set to %.3f.", scale)

def load_lora_adapter(pipe, lora_config):
    if not lora_config['enabled']:
        logger.info("LoRA is disabled in config.")
        return

    lora_path = lora_config['path']
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    adapter_name = lora_config['adapter_name']
    scale = lora_config['scale']
    logger.info("Loading LoRA adapter %s from %s at scale %.3f...", adapter_name, lora_path, scale)

    raw_state_dict = load_file(str(lora_path), device='cpu')
    state_dict, network_alphas, metadata = StableDiffusionPipeline.lora_state_dict(
        dict(raw_state_dict),
        return_lora_metadata=True,
    )

    if lora_config['load_unet']:
        unet_state_dict = {k: v for k, v in state_dict.items() if k.startswith('unet.')}
        unet_network_alphas = {
            k: v for k, v in (network_alphas or {}).items() if k.startswith('unet.')
        }
        if not unet_state_dict:
            raise ValueError(f"No UNet LoRA tensors found in {lora_path}")

        StableDiffusionPipeline.load_lora_into_unet(
            unet_state_dict,
            unet_network_alphas,
            pipe.unet,
            adapter_name=adapter_name,
            _pipeline=pipe,
            metadata=metadata,
        )
        pipe.set_adapters(adapter_name, adapter_weights=0.0)
        logger.info("Loaded %s UNet LoRA tensor(s).", len(unet_state_dict))

    if lora_config['fuse'] and lora_config['load_unet']:
        logger.info("Fusing LoRA adapter %s at scale %.3f.", adapter_name, scale)
        pipe.fuse_lora(components=['unet'], lora_scale=scale, adapter_names=[adapter_name])

    del state_dict
    gc.collect()

    handle = LoraAdapterHandle(
        adapter_name=adapter_name,
        raw_state_dict=raw_state_dict,
        load_unet=lora_config['load_unet'],
        load_text_encoder=lora_config['load_text_encoder'],
    )
    handle.set_scale(pipe, scale)
    logger.info("LoRA adapter loaded. RAM: %.2f MB", get_ram_usage())
    return handle

def prompt_text(prompt):
    if isinstance(prompt, dict):
        prompt = (
            prompt.get('prompt')
            or prompt.get('text')
            or prompt.get('scene_prompt')
            or json.dumps(prompt)
        )
    return str(prompt).strip()

def prompt_id(prompt, image_index):
    if isinstance(prompt, dict):
        return prompt.get('id') or prompt.get('prompt_id') or f"prompt_{image_index:06d}"
    return f"prompt_{image_index:06d}"

def prompt_scene(prompt):
    if isinstance(prompt, dict):
        scene = prompt.get('scene_prompt')
        if scene:
            return str(scene).strip()

    text = prompt_text(prompt)
    return text.replace("<tlqueen>", "", 1).strip(" ,")

def build_training_caption(prompt, character_prompt):
    if isinstance(prompt, dict) and prompt.get('training_caption'):
        return str(prompt['training_caption']).strip()

    parts = ["<tlqueen>"]
    if character_prompt and character_prompt.strip():
        parts.append(character_prompt.strip())
    scene = prompt_scene(prompt)
    if scene:
        parts.append(scene)
    return ", ".join(parts)

def build_prompt(prompt, style_prompt, character_prompt):
    prompt = prompt_text(prompt)
    prompt = prompt.strip()
    character_prompt = character_prompt.strip() if character_prompt else ""
    if character_prompt:
        trigger = "<tlqueen>"
        if trigger in prompt:
            prompt = prompt.replace(trigger, f"single {trigger}, {character_prompt},", 1)
        else:
            prompt = f"single character, {character_prompt}, {prompt}"

    prefix_parts = [
        part.strip()
        for part in (style_prompt,)
        if part and part.strip()
    ]
    if not prefix_parts:
        return prompt
    return f"{prompt}, {', '.join(prefix_parts)}"

def get_dataset_config(config):
    dataset_config = config.get('dataset', {})
    return {
        'write_caption_files': bool(dataset_config.get('write_caption_files', False)),
        'caption_extension': str(dataset_config.get('caption_extension', '.txt')),
    }

def load_manifest(config, config_path, generation_config, lora_config, two_pass_config):
    manifest_file = config.get('manifest_file')
    if not manifest_file:
        return None, None

    manifest_path = Path(manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parent / manifest_path).resolve()

    if manifest_path.exists():
        with manifest_path.open('r', encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        manifest = {
            'version': 1,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'config_file': str(config_path),
            'prompt_file': config.get('prompts_file'),
            'records': [],
        }

    manifest.setdefault('version', 1)
    manifest.setdefault('records', [])
    manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['generation'] = {
        'width': generation_config['width'],
        'height': generation_config['height'],
        'style_prompt': generation_config['style_prompt'],
        'character_prompt': generation_config['character_prompt'],
        'negative_prompt': generation_config['negative_prompt'],
        'seed': generation_config['seed'],
    }
    manifest['lora'] = {
        'path': str(lora_config['path']),
        'scale': lora_config['scale'],
        'adapter_name': lora_config['adapter_name'],
    }
    manifest['two_pass'] = two_pass_config
    return manifest_path, manifest

def save_manifest(manifest_path, manifest):
    if manifest_path is None or manifest is None:
        return

    manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')
    os.makedirs(manifest_path.parent, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)

def caption_path_for_image(image_path, caption_extension):
    if not caption_extension.startswith('.'):
        caption_extension = f".{caption_extension}"
    return str(Path(image_path).with_suffix(caption_extension))

def write_caption_file(image_path, caption, dataset_config):
    if not dataset_config['write_caption_files']:
        return None

    caption_path = caption_path_for_image(image_path, dataset_config['caption_extension'])
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(caption.strip() + "\n")
    return caption_path

def monster_color_for_index(image_index):
    colors = [
        (83, 190, 104),
        (151, 96, 210),
        (89, 155, 226),
        (230, 139, 67),
        (80, 185, 176),
        (145, 145, 150),
        (232, 170, 202),
        (84, 176, 95),
        (126, 83, 54),
        (95, 142, 210),
    ]
    return colors[(image_index - 1) % len(colors)]

def create_single_subject_draft(width, height, image_index):
    image = Image.new("RGB", (width, height), (182, 220, 235))
    draw = ImageDraw.Draw(image)

    # Simple deterministic layout: one princess-shaped subject, one non-human target.
    ground_y = int(height * 0.78)
    draw.rectangle((0, ground_y, width, height), fill=(134, 207, 132))
    draw.ellipse((-int(width * 0.2), -int(height * 0.3), int(width * 0.55), int(height * 0.55)), fill=(205, 235, 244))

    queen_x = int(width * 0.38)
    head_y = int(height * 0.30)
    head_r = max(18, int(min(width, height) * 0.08))
    dress_top_y = head_y + head_r + 6
    dress_bottom_y = int(height * 0.72)
    dress_half_w = int(width * 0.11)

    hair_color = (126, 80, 150)
    hair_shadow = (92, 55, 120)
    skin_color = (246, 210, 184)
    pink = (239, 109, 176)
    light_pink = (255, 166, 210)
    dark_pink = (194, 70, 141)
    gold = (235, 190, 55)
    green_eye = (32, 130, 94)

    draw.ellipse((queen_x - head_r - 10, head_y - head_r - 2, queen_x + head_r + 10, head_y + head_r + 18), fill=hair_shadow)
    draw.ellipse((queen_x - head_r - 6, head_y - head_r - 2, queen_x + head_r + 6, head_y + head_r + 10), fill=hair_color)
    draw.pieslice((queen_x - head_r - 30, head_y, queen_x - 4, head_y + head_r * 4), 75, 240, fill=hair_color)
    draw.pieslice((queen_x + 4, head_y, queen_x + head_r + 30, head_y + head_r * 4), -60, 105, fill=hair_color)
    draw.ellipse((queen_x - head_r, head_y - head_r, queen_x + head_r, head_y + head_r), fill=skin_color)
    draw.polygon(
        [
            (queen_x - int(head_r * 0.9), head_y - head_r - 2),
            (queen_x - int(head_r * 0.4), head_y - head_r - int(head_r * 0.85)),
            (queen_x, head_y - head_r - 2),
            (queen_x + int(head_r * 0.45), head_y - head_r - int(head_r * 0.9)),
            (queen_x + int(head_r * 0.9), head_y - head_r - 2),
        ],
        fill=gold,
    )
    jewel_r = max(2, head_r // 8)
    draw.ellipse((queen_x - jewel_r, head_y - head_r - int(head_r * 0.45) - jewel_r, queen_x + jewel_r, head_y - head_r - int(head_r * 0.45) + jewel_r), fill=(74, 207, 120))
    draw.ellipse((queen_x - int(head_r * 0.48) - jewel_r, head_y - head_r - int(head_r * 0.25) - jewel_r, queen_x - int(head_r * 0.48) + jewel_r, head_y - head_r - int(head_r * 0.25) + jewel_r), fill=(239, 105, 170))
    draw.ellipse((queen_x + int(head_r * 0.48) - jewel_r, head_y - head_r - int(head_r * 0.25) - jewel_r, queen_x + int(head_r * 0.48) + jewel_r, head_y - head_r - int(head_r * 0.25) + jewel_r), fill=(239, 105, 170))
    eye_r = max(2, head_r // 9)
    draw.ellipse((queen_x - head_r // 2, head_y - 2, queen_x - head_r // 2 + eye_r * 2, head_y + eye_r * 2), fill=green_eye)
    draw.ellipse((queen_x + head_r // 3, head_y - 2, queen_x + head_r // 3 + eye_r * 2, head_y + eye_r * 2), fill=green_eye)
    draw.arc((queen_x - 6, head_y + 8, queen_x + 8, head_y + 18), 15, 165, fill=(120, 45, 70), width=2)

    draw.polygon(
        [
            (queen_x, dress_top_y),
            (queen_x - dress_half_w, dress_bottom_y),
            (queen_x + dress_half_w, dress_bottom_y),
        ],
        fill=pink,
        outline=dark_pink,
    )
    draw.ellipse((queen_x - dress_half_w - 16, dress_top_y - 2, queen_x - dress_half_w + 12, dress_top_y + 28), fill=light_pink, outline=dark_pink)
    draw.ellipse((queen_x + dress_half_w - 12, dress_top_y - 2, queen_x + dress_half_w + 16, dress_top_y + 28), fill=light_pink, outline=dark_pink)
    draw.arc((queen_x - dress_half_w, dress_bottom_y - 20, queen_x + dress_half_w, dress_bottom_y + 18), 0, 180, fill=dark_pink, width=3)
    draw.ellipse((queen_x - 10, dress_top_y + 12, queen_x + 10, dress_top_y + 32), fill=gold)
    arm_y = dress_top_y + int((dress_bottom_y - dress_top_y) * 0.2)
    draw.line((queen_x - 8, arm_y, queen_x - dress_half_w - 36, arm_y + 12), fill=skin_color, width=5)
    wand_start = (queen_x + 10, arm_y)
    wand_end = (int(width * 0.73), int(height * 0.39))
    draw.line((queen_x + 8, arm_y, *wand_end), fill=gold, width=5)
    draw.line((queen_x - 12, dress_bottom_y, queen_x - 18, int(height * 0.88)), fill=skin_color, width=5)
    draw.line((queen_x + 12, dress_bottom_y, queen_x + 18, int(height * 0.88)), fill=skin_color, width=5)

    star_x, star_y = wand_end
    star_r = max(10, int(min(width, height) * 0.035))
    draw.polygon(
        [
            (star_x, star_y - star_r),
            (star_x + star_r // 3, star_y - star_r // 3),
            (star_x + star_r, star_y),
            (star_x + star_r // 3, star_y + star_r // 3),
            (star_x, star_y + star_r),
            (star_x - star_r // 3, star_y + star_r // 3),
            (star_x - star_r, star_y),
            (star_x - star_r // 3, star_y - star_r // 3),
        ],
        fill=(255, 242, 75),
    )

    monster_x = int(width * 0.82)
    monster_y = int(height * 0.64)
    monster_w = int(width * 0.12)
    monster_h = int(height * 0.13)
    monster_color = monster_color_for_index(image_index)
    draw.ellipse(
        (monster_x - monster_w, monster_y - monster_h, monster_x + monster_w, monster_y + monster_h),
        fill=monster_color,
        outline=(60, 100, 60),
        width=3,
    )
    draw.ellipse((monster_x - 18, monster_y - 12, monster_x - 10, monster_y - 4), fill=(20, 20, 20))
    draw.ellipse((monster_x + 10, monster_y - 12, monster_x + 18, monster_y - 4), fill=(20, 20, 20))
    draw.arc((monster_x - 20, monster_y - 4, monster_x + 20, monster_y + 20), 10, 170, fill=(20, 20, 20), width=2)
    draw.line((star_x, star_y, monster_x - monster_w, monster_y), fill=(255, 245, 112), width=4)

    return image

def existing_images_for_index(output_dir, index):
    exact_path = os.path.join(output_dir, f"image_{index}.png")
    timestamped_paths = glob.glob(os.path.join(output_dir, f"image_{index}_*.png"))
    paths = []
    if os.path.exists(exact_path):
        paths.append(exact_path)
    paths.extend(timestamped_paths)
    return paths

def generator_for_image(seed, image_index, pass_offset=0):
    if seed is None:
        return None
    return torch.Generator(device="cpu").manual_seed(seed + image_index - 1 + pass_offset)

def generate_images(config_path):
    config_path = resolve_config_path(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    resolve_config_file_paths(config, config_path)
    generation_config = get_generation_config(config)
    lora_config = get_lora_config(config, config_path)
    two_pass_config = get_two_pass_config(config, generation_config, lora_config)
    upscale_config = get_upscale_config(config)
    dataset_config = get_dataset_config(config)
    manifest_path, manifest = load_manifest(
        config,
        config_path,
        generation_config,
        lora_config,
        two_pass_config,
    )

    with open(config['prompts_file'], 'r') as f:
        prompts = json.load(f)
    prompt_count = int(config.get('prompt_count', len(prompts)))
    target_prompts = prompts[:prompt_count]
    if prompt_count > len(prompts):
        logger.warning("Requested %s images but prompts file only has %s prompts.", prompt_count, len(prompts))

    logger.info(f"Initial RAM: {get_ram_usage():.2f} MB")
    logger.info(f"local_files_only={generation_config['local_files_only']}")

    # Load SD 1.5 components sequentially
    model_id = os.environ.get("SD15_MODEL", "runwayml/stable-diffusion-v1-5")
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
    lora_handle = load_lora_adapter(pipe, lora_config)
    img2img_pipe = None
    if two_pass_config['enabled']:
        logger.info("Creating StableDiffusionImg2ImgPipeline with shared components...")
        img2img_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        img2img_pipe.to("cpu")

    logger.info(f"RAM after pipeline load: {get_ram_usage():.2f} MB")
    logger.info(
        "Generation settings: %sx%s, %s steps, guidance_scale=%s, seed=%s, two_pass=%s",
        generation_config['width'],
        generation_config['height'],
        generation_config['num_inference_steps'],
        generation_config['guidance_scale'],
        generation_config['seed'],
        two_pass_config['enabled'],
    )
    if two_pass_config['enabled']:
        logger.info(
            "Two-pass settings: draft scale=%.3f, draft steps=%s, refine scale=%.3f, refine steps=%s, strength=%.2f",
            two_pass_config['draft_lora_scale'],
            two_pass_config['draft_num_inference_steps'],
            two_pass_config['refine_lora_scale'],
            two_pass_config['refine_num_inference_steps'],
            two_pass_config['strength'],
        )

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['upscaled_dir'], exist_ok=True)
    drafts_dir = os.path.join(config['output_dir'], "drafts")
    if two_pass_config['enabled'] and two_pass_config['save_drafts']:
        os.makedirs(drafts_dir, exist_ok=True)
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

            final_prompt = build_prompt(
                prompt,
                generation_config['style_prompt'],
                generation_config['character_prompt'],
            )
            logger.info(f"Generating image {i+1}/{len(target_prompts)}: {final_prompt[:80]}...")

            if two_pass_config['enabled']:
                draft_path = None
                if two_pass_config['draft_source'] == 'template':
                    draft_image = create_single_subject_draft(
                        generation_config['width'],
                        generation_config['height'],
                        image_index,
                    )
                elif two_pass_config['draft_source'] == 'generated':
                    lora_handle.set_scale(pipe, two_pass_config['draft_lora_scale'])
                    draft_image = pipe(
                        final_prompt,
                        negative_prompt=generation_config['negative_prompt'],
                        num_inference_steps=two_pass_config['draft_num_inference_steps'],
                        guidance_scale=two_pass_config['draft_guidance_scale'],
                        height=generation_config['height'],
                        width=generation_config['width'],
                        generator=generator_for_image(generation_config['seed'], image_index),
                    ).images[0]
                else:
                    raise ValueError(f"Unsupported two_pass.draft_source: {two_pass_config['draft_source']}")

                if two_pass_config['save_drafts']:
                    draft_filename = f"draft_image_{i+1}_{timestamp}.png"
                    draft_path = os.path.join(drafts_dir, draft_filename)
                    draft_image.save(draft_path)

                lora_handle.set_scale(pipe, two_pass_config['refine_lora_scale'])
                image = img2img_pipe(
                    final_prompt,
                    image=draft_image,
                    strength=two_pass_config['strength'],
                    negative_prompt=generation_config['negative_prompt'],
                    num_inference_steps=two_pass_config['refine_num_inference_steps'],
                    guidance_scale=two_pass_config['refine_guidance_scale'],
                    generator=generator_for_image(generation_config['seed'], image_index, pass_offset=10000),
                ).images[0]
                del draft_image
            else:
                draft_path = None
                image = pipe(
                    final_prompt,
                    negative_prompt=generation_config['negative_prompt'],
                    num_inference_steps=generation_config['num_inference_steps'],
                    guidance_scale=generation_config['guidance_scale'],
                    height=generation_config['height'],
                    width=generation_config['width'],
                    generator=generator_for_image(generation_config['seed'], image_index),
                ).images[0]

            output_filename = f"image_{i+1}_{timestamp}.png"
            input_path = os.path.join(config['output_dir'], output_filename)
            output_path = os.path.join(config['upscaled_dir'], output_filename)
            image.save(input_path)
            training_caption = build_training_caption(
                prompt,
                generation_config['character_prompt'],
            )
            caption_path = write_caption_file(input_path, training_caption, dataset_config)
            pending_upscales.append(
                upscale_executor.submit(upscale_file, input_path, output_path, upscale_config)
            )
            if manifest is not None:
                manifest['records'].append({
                    'run_id': timestamp,
                    'prompt_id': prompt_id(prompt, image_index),
                    'image_index': image_index,
                    'scene_prompt': prompt_scene(prompt),
                    'prompt': prompt_text(prompt),
                    'training_caption': training_caption,
                    'final_prompt': final_prompt,
                    'negative_prompt': generation_config['negative_prompt'],
                    'draft_file': draft_path,
                    'image_file': input_path,
                    'upscaled_file': output_path,
                    'caption_file': caption_path,
                    'width': generation_config['width'],
                    'height': generation_config['height'],
                    'created_at': datetime.now().isoformat(timespec='seconds'),
                })
                save_manifest(manifest_path, manifest)

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
        save_manifest(manifest_path, manifest)

    # Unload
    if img2img_pipe is not None:
        del img2img_pipe
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
    generate_images(sys.argv[1] if len(sys.argv) > 1 else "config_lora_littlequeen.json")
