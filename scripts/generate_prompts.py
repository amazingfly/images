
# Load centralized workstation defaults; explicit environment/CLI values win.
import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
for _workspace_root in _WorkspacePath(__file__).resolve().parents:
    if (_workspace_root / "media_workspace").is_dir():
        _workspace_sys.path.insert(0, str(_workspace_root))
        break
from media_workspace.config import apply_environment as _apply_workspace
_apply_workspace()

import gc
import gzip
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psutil

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


def log_memory(label):
    vm = psutil.virtual_memory()
    logger.info(
        "%s. System available: %.2f GB, used: %.1f%%",
        label,
        vm.available / (1024 ** 3),
        vm.percent,
    )


def get_prompt_generation_config(config):
    prompt_config = config.get('prompt_generation', {})
    prompt_count = int(config.get('prompt_count', 50))
    batch_size = max(1, min(int(prompt_config.get('batch_size', 1)), prompt_count))
    default_attempts = max(10, math.ceil(prompt_count / batch_size) * 3)

    return {
        'server_binary': Path(prompt_config.get(
            'server_binary',
            '/home/derek/projects/hug/gemma4_uncensored/llama.cpp/build/bin/llama-server',
        )),
        'model_path': Path(prompt_config.get(
            'model_path',
            '/home/derek/projects/hug/gemma4_uncensored/models/gemma4/'
            'Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf',
        )),
        'host': str(prompt_config.get('host', '127.0.0.1')),
        'port': int(prompt_config.get('port', 18080)),
        'threads': int(prompt_config.get('threads', 6)),
        'context_size': int(prompt_config.get('context_size', 8192)),
        'batch_size': batch_size,
        'llama_batch_size': int(prompt_config.get('llama_batch_size', 512)),
        'llama_ubatch_size': int(prompt_config.get('llama_ubatch_size', 128)),
        'cache_ram_mb': int(prompt_config.get('cache_ram_mb', 256)),
        'startup_timeout_seconds': int(prompt_config.get('startup_timeout_seconds', 180)),
        'request_timeout_seconds': int(prompt_config.get('request_timeout_seconds', 1800)),
        'max_attempts': int(prompt_config.get('max_attempts', default_attempts)),
        'max_new_tokens_per_prompt': int(prompt_config.get('max_new_tokens_per_prompt', 80)),
        'max_new_tokens_cap': int(prompt_config.get('max_new_tokens_cap', 160)),
        'temperature': float(prompt_config.get('temperature', 0.9)),
        'top_p': float(prompt_config.get('top_p', 0.95)),
        'minimum_words': int(prompt_config.get('minimum_words', 18)),
        'maximum_words': int(prompt_config.get('maximum_words', 28)),
        'similarity_limit': float(prompt_config.get('similarity_limit', 0.82)),
    }


def archive_old_prompts(prompts_file):
    if os.path.exists(prompts_file):
        archive_dir = os.path.join(os.path.dirname(prompts_file), "oldPrompts")
        os.makedirs(archive_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(archive_dir, f"prompts_{timestamp}.json.gz")

        logger.info("Archiving old prompts to %s...", archive_path)
        with open(prompts_file, 'rb') as f_in:
            with gzip.open(archive_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info("Archiving complete.")


def write_prompts(prompts_file, prompts):
    archive_old_prompts(prompts_file)

    tmp_path = f"{prompts_file}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(prompts, f, indent=4)
    os.replace(tmp_path, prompts_file)


def normalize_prompt(prompt):
    prompt = prompt.strip()
    prompt = re.sub(r'^```(?:text)?\s*|\s*```$', '', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'^\s*(?:prompt\s*:\s*|\d+[.)]\s*)', '', prompt, flags=re.IGNORECASE)
    prompt = prompt.strip().strip('"').strip("'").strip()
    return re.sub(r'\s+', ' ', prompt)


def word_count(prompt):
    return len(re.findall(r"\b[\w'-]+\b", prompt))


def prompts_are_similar(left, right, similarity_limit):
    left_normalized = normalize_prompt(left).lower()
    right_normalized = normalize_prompt(right).lower()
    return SequenceMatcher(None, left_normalized, right_normalized).ratio() >= similarity_limit


def validate_prompt(prompt, existing_prompts, prompt_config):
    prompt = normalize_prompt(prompt)
    words = word_count(prompt)
    if words < prompt_config['minimum_words']:
        return None, f"too short ({words} words)"
    if words > prompt_config['maximum_words']:
        return None, f"too long ({words} words)"

    lower = prompt.lower()
    forbidden_phrases = (
        'return only',
        'numbered list',
        'image prompts',
        'style boilerplate',
        'camera-distance',
        'token limit',
        'avoid repeating',
        'desired output',
        'here is',
    )
    leaked = next((phrase for phrase in forbidden_phrases if phrase in lower), None)
    if leaked:
        return None, f"contains instruction text: {leaked}"
    if '\n' in prompt:
        return None, "contains multiple lines"
    if any(prompts_are_similar(prompt, existing, prompt_config['similarity_limit']) for existing in existing_prompts):
        return None, "too similar to an existing prompt"
    return prompt, None


def build_system_prompt(prompt_config):
    return (
        "You write one concise content prompt for Stable Diffusion 1.5. SD1.5 CLIP has "
        "a hard 77-token context, and a separate style prefix already consumes part of "
        "that limit. Output exactly one sentence of "
        f"{prompt_config['minimum_words']} to {prompt_config['maximum_words']} words, "
        "roughly no more than 35 CLIP tokens. Describe one clearly visible primary "
        "character, a specific pose or action, one distinctive weapon or cybernetic "
        "feature, and a concrete environment. Put the most important subject and action "
        "first. Use literal visual nouns and active verbs. Do not include art style, "
        "camera distance, quality tags, negative instructions, explanations, labels, "
        "quotes, JSON, numbering, or multiple alternatives.\n\n"
        "Desired output example:\n"
        "A chrome-armed cyborg duelist vaults across a rain-slick maglev platform, "
        "plasma spear raised, neon signs reflecting through smoke and flying debris."
    )


def build_prompt_request(config, existing_prompts=None):
    prompt_request = str(config.get('prompt_request') or "").strip()
    if not prompt_request:
        prompt_request = (
            "Create a dystopian cyborg woman in combat with a futuristic weapon in a "
            "colorful neon environment."
        )

    duplicate_note = ""
    if existing_prompts:
        duplicate_note = (
            "\nMake this concept substantially different from these recent prompts:\n"
            + "\n".join(f"- {prompt}" for prompt in existing_prompts[-8:])
        )

    return (
        f"Create one new image prompt based on this request:\n{prompt_request}"
        f"{duplicate_note}"
    )


def http_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(url, data=data)
    request.add_header('Content-Type', 'application/json')
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"Gemma 4 server returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Gemma 4 server at {url}: {exc}") from exc


def start_llama_server(prompt_config):
    binary = prompt_config['server_binary']
    model = prompt_config['model_path']
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Missing executable llama-server: {binary}")
    if not model.is_file():
        raise FileNotFoundError(f"Missing Gemma 4 Q5 model: {model}")

    host = prompt_config['host']
    port = prompt_config['port']
    health_url = f"http://{host}:{port}/health"
    log_path = Path(f"/tmp/sd15-gemma4-server-{os.getpid()}.log")
    log_file = log_path.open('w', encoding='utf-8')
    env = os.environ.copy()
    binary_dir = str(binary.parent)
    env['LD_LIBRARY_PATH'] = f"{binary_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault('MALLOC_ARENA_MAX', '2')

    command = [
        str(binary),
        '-m', str(model),
        '--host', host,
        '--port', str(port),
        '--threads', str(prompt_config['threads']),
        '--threads-batch', str(prompt_config['threads']),
        '--ctx-size', str(prompt_config['context_size']),
        '--parallel', '1',
        '--n-gpu-layers', '0',
        '--reasoning', 'off',
        '--reasoning-budget', '0',
        '--timeout', str(prompt_config['request_timeout_seconds']),
        '--batch-size', str(prompt_config['llama_batch_size']),
        '--ubatch-size', str(prompt_config['llama_ubatch_size']),
        '--cache-ram', str(prompt_config['cache_ram_mb']),
    ]

    logger.info("Starting Gemma 4 E4B Q5 text server. Log: %s", log_path)
    logger.info("Model: %s", model)
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    deadline = time.monotonic() + prompt_config['startup_timeout_seconds']
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.close()
            details = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
            raise RuntimeError(f"Gemma 4 server exited during startup:\n{details}")
        try:
            health = http_json(health_url, timeout=2)
            if health.get('status') == 'ok':
                log_memory("Gemma 4 E4B Q5 server ready")
                return process, log_file, log_path
        except RuntimeError:
            pass
        time.sleep(1)

    stop_llama_server(process, log_file)
    details = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
    raise TimeoutError(f"Gemma 4 server did not become ready:\n{details}")


def stop_llama_server(process, log_file):
    if process.poll() is None:
        logger.info("Stopping Gemma 4 server before SD1.5 starts...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    if not log_file.closed:
        log_file.close()
    gc.collect()
    log_memory("Gemma 4 server stopped")


def generate_prompt(prompt_config, system_prompt, request_text, max_tokens):
    url = f"http://{prompt_config['host']}:{prompt_config['port']}/v1/chat/completions"
    response = http_json(
        url,
        payload={
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': request_text},
            ],
            'temperature': prompt_config['temperature'],
            'top_p': prompt_config['top_p'],
            'max_tokens': max_tokens,
        },
        timeout=prompt_config['request_timeout_seconds'],
    )
    try:
        return response['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemma 4 response: {response}") from exc


def generate_prompts(config_path):
    config_path = resolve_config_path(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    resolve_config_file_paths(config, config_path)
    prompt_count = int(config.get('prompt_count', 50))
    prompt_config = get_prompt_generation_config(config)

    log_memory("Starting Gemma 4 prompt generation")
    process = None
    log_file = None
    prompts = []
    try:
        process, log_file, _ = start_llama_server(prompt_config)
        system_prompt = build_system_prompt(prompt_config)
        for attempt in range(1, prompt_config['max_attempts'] + 1):
            remaining = prompt_count - len(prompts)
            if remaining <= 0:
                break

            request_text = build_prompt_request(config, prompts)
            max_tokens = max(
                96,
                min(
                    prompt_config['max_new_tokens_cap'],
                    prompt_config['max_new_tokens_per_prompt'],
                ),
            )
            logger.info(
                "Requesting one prompt, attempt %s/%s: need %s more...",
                attempt,
                prompt_config['max_attempts'],
                remaining,
            )
            response = generate_prompt(prompt_config, system_prompt, request_text, max_tokens)
            prompt, rejection_reason = validate_prompt(response, prompts, prompt_config)
            if prompt is None:
                logger.warning("Rejected Gemma 4 response: %s", rejection_reason)
                continue
            prompts.append(prompt)
            logger.info("Accepted prompt %s/%s: %s", len(prompts), prompt_count, prompt)
    finally:
        if process is not None and log_file is not None:
            stop_llama_server(process, log_file)

    final_prompts = prompts[:prompt_count]
    if len(final_prompts) < prompt_count:
        raise RuntimeError(
            f"Requested {prompt_count} prompts but Gemma 4 only produced "
            f"{len(final_prompts)} unique parseable prompts."
        )

    write_prompts(config['prompts_file'], final_prompts)
    logger.info("Generated and saved %s prompts.", len(final_prompts))


if __name__ == "__main__":
    generate_prompts(sys.argv[1] if len(sys.argv) > 1 else "config.json")
