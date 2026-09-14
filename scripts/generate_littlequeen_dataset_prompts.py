import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from generate_prompts import (
    generate_prompt,
    get_prompt_generation_config,
    log_memory,
    normalize_prompt,
    prompts_are_similar,
    resolve_config_file_paths,
    resolve_config_path,
    start_llama_server,
    stop_llama_server,
    word_count,
    write_prompts,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FORBIDDEN_TERMS = (
    '<tlqueen>',
    'tlqueen',
    'queen',
    'princess',
    'girl',
    'child',
    'kid',
    'woman',
    'female',
    'character',
    'face',
    'hair',
    'eyes',
    'crown',
    'necklace',
    'pendant',
    'chestnut',
    'brown eyes',
    'pink dress',
    'magenta dress',
    'anime',
    'storybook',
    'disney',
    'pixar',
    'cartoon',
    'cel shading',
    'cel-shading',
    'line art',
    'render',
    'quality',
    'detailed',
    'high detail',
    'cinematic',
    'camera',
    'portrait',
    'close-up',
    'style',
    'lighting',
    'palette',
)


def build_system_prompt(prompt_config):
    return (
        "You write short scene/action fragments for a Stable Diffusion 1.5 LoRA dataset.\n"
        "A separate pipeline already adds the fixed character, fixed art style, and negative prompt.\n"
        "Return only a valid JSON array of strings. No markdown, labels, numbering, or explanation.\n"
        f"Each string must be {prompt_config['minimum_words']} to {prompt_config['maximum_words']} words.\n"
        "Describe only scene, action, prop, simple monster, setting, or accessory.\n"
        "Do not describe the person, identity, face, hair, eyes, crown, age, gender, or art style.\n"
        "Do not include <tlqueen>, queen, princess, anime, storybook, Disney, quality tags, or camera words.\n"
        "Good examples:\n"
        "[\n"
        '  "blasting a gumdrop goblin with a ruby wand",\n'
        '  "shielding a tiny dragon beside candy mushrooms",\n'
        '  "summoning star sparks at a toy slime monster",\n'
        '  "holding a moon scepter near a sleepy ogre",\n'
        '  "riding a carousel horse through floating bubbles"\n'
        "]"
    )


def build_user_request(config, batch_count, existing_prompts):
    theme = str(config.get('prompt_request') or "").strip()
    if not theme:
        theme = (
            "playful fairytale magic scenes with harmless childish monsters, toys, gems, "
            "wands, scepters, candy, flowers, castles, and simple action"
        )

    recent = existing_prompts[-12:]
    duplicate_note = ""
    if recent:
        duplicate_note = (
            "\nAvoid repeating these existing fragments:\n"
            + "\n".join(f"- {prompt}" for prompt in recent)
        )

    return (
        f"Create exactly {batch_count} new unique scene fragments for this theme:\n"
        f"{theme}\n"
        "Keep each fragment short and literal. Focus on the action and scene object only."
        f"{duplicate_note}"
    )


def strip_json_fence(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def parse_prompt_items(response_text):
    text = strip_json_fence(response_text)
    candidates = [text]

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, str)]

    items = []
    for line in text.splitlines():
        line = normalize_scene_fragment(line)
        if line:
            items.append(line)
    return items


def normalize_scene_fragment(fragment):
    fragment = normalize_prompt(fragment)
    fragment = re.sub(r'^\s*[-*]\s*', '', fragment)
    fragment = re.sub(r'^\s*["\']|["\'],?\s*$', '', fragment)
    fragment = fragment.strip().strip('[]').strip()
    fragment = fragment.rstrip('.,;:')
    return re.sub(r'\s+', ' ', fragment)


def validate_scene_fragment(fragment, existing_prompts, prompt_config):
    fragment = normalize_scene_fragment(fragment)
    if not fragment:
        return None, "empty"
    if '\n' in fragment:
        return None, "contains multiple lines"

    words = word_count(fragment)
    if words < prompt_config['minimum_words']:
        return None, f"too short ({words} words)"
    if words > prompt_config['maximum_words']:
        return None, f"too long ({words} words)"

    lower = fragment.lower()
    leaked = next((term for term in FORBIDDEN_TERMS if term in lower), None)
    if leaked:
        return None, f"contains forbidden term: {leaked}"

    if any(prompts_are_similar(fragment, existing, prompt_config['similarity_limit']) for existing in existing_prompts):
        return None, "too similar to an existing prompt"

    return fragment, None


def build_training_caption(scene_prompt, character_prompt):
    parts = ["<tlqueen>"]
    if character_prompt:
        parts.append(character_prompt.strip())
    parts.append(scene_prompt)
    return ", ".join(part for part in parts if part)


def build_record(index, scene_prompt, config, prompt_config, created_at):
    character_prompt = config.get('generation', {}).get('character_prompt', '')
    return {
        'id': f"lq_scene_{index:06d}",
        'scene_prompt': scene_prompt,
        'prompt': f"<tlqueen> {scene_prompt}",
        'training_caption': build_training_caption(scene_prompt, character_prompt),
        'source': {
            'type': 'gemma4_e4b_llama_cpp_server',
            'model_path': str(prompt_config['model_path']),
        },
        'created_at': created_at,
    }


def generate_dataset_prompts(config_path):
    config_path = resolve_config_path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    resolve_config_file_paths(config, config_path)

    prompt_count = int(config.get('prompt_count', 50))
    prompt_config = get_prompt_generation_config(config)
    batch_size = max(1, min(int(prompt_config.get('batch_size', 1)), prompt_count))
    max_attempts = int(prompt_config.get('max_attempts', max(10, math.ceil(prompt_count / batch_size) * 3)))
    created_at = datetime.now().isoformat(timespec='seconds')

    records = []
    accepted_scene_prompts = []
    process = None
    log_file = None
    log_memory("Starting Little Queen Gemma scene prompt generation")
    try:
        process, log_file, _ = start_llama_server(prompt_config)
        system_prompt = build_system_prompt(prompt_config)
        for attempt in range(1, max_attempts + 1):
            remaining = prompt_count - len(records)
            if remaining <= 0:
                break

            request_count = min(batch_size, remaining)
            request_text = build_user_request(config, request_count, accepted_scene_prompts)
            max_tokens = max(
                128,
                min(
                    int(prompt_config['max_new_tokens_cap']),
                    request_count * int(prompt_config['max_new_tokens_per_prompt']) + 64,
                ),
            )
            logger.info(
                "Requesting %s scene prompt(s), attempt %s/%s: need %s more...",
                request_count,
                attempt,
                max_attempts,
                remaining,
            )
            response = generate_prompt(prompt_config, system_prompt, request_text, max_tokens)
            candidates = parse_prompt_items(response)
            if not candidates:
                logger.warning("Rejected Gemma response: could not parse prompt array")
                continue

            for candidate in candidates:
                fragment, rejection_reason = validate_scene_fragment(
                    candidate,
                    accepted_scene_prompts,
                    prompt_config,
                )
                if fragment is None:
                    logger.warning("Rejected scene fragment %r: %s", candidate, rejection_reason)
                    continue

                accepted_scene_prompts.append(fragment)
                records.append(
                    build_record(
                        len(records) + 1,
                        fragment,
                        config,
                        prompt_config,
                        created_at,
                    )
                )
                logger.info("Accepted scene prompt %s/%s: %s", len(records), prompt_count, fragment)
                if len(records) >= prompt_count:
                    break
    finally:
        if process is not None and log_file is not None:
            stop_llama_server(process, log_file)

    if len(records) < prompt_count:
        raise RuntimeError(
            f"Requested {prompt_count} prompts but Gemma only produced "
            f"{len(records)} unique valid scene prompts."
        )

    os.makedirs(os.path.dirname(config['prompts_file']), exist_ok=True)
    write_prompts(config['prompts_file'], records[:prompt_count])
    logger.info("Generated and saved %s Little Queen dataset scene prompts.", len(records[:prompt_count]))
    return records[:prompt_count]


if __name__ == "__main__":
    generate_dataset_prompts(sys.argv[1] if len(sys.argv) > 1 else "config_lora_littlequeen_dataset.json")
