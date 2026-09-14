import sys
from pathlib import Path

import pytest

from run_sdxl_cpu import build_command, validate_inputs


def backend_config(tmp_path):
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'fixture')
    return {'binary': sys.executable, 'model': str(model), 'lora_dir': str(tmp_path),
            'generation': {'width': 512, 'height': 768, 'steps': 20, 'cfg_scale': 7,
                'sampling_method': 'euler', 'scheduler': 'normal', 'threads': 2, 'rng': 'cpu'}}


def test_backend_rejects_missing_model_before_generation(tmp_path):
    config = backend_config(tmp_path)
    Path(config['model']).unlink()
    with pytest.raises(FileNotFoundError, match='quantized SDXL model'):
        validate_inputs(config)


def test_backend_checks_per_scene_loras(tmp_path):
    config = backend_config(tmp_path)
    config['prompts'] = [{'loras': [{'name': 'scene_identity', 'weight': 0.7}]}]
    with pytest.raises(FileNotFoundError, match='scene_identity'):
        validate_inputs(config)
    (tmp_path / 'scene_identity.safetensors').write_bytes(b'fixture')
    validate_inputs(config)


def test_scene_lora_override_does_not_mutate_shared_config(tmp_path):
    config = backend_config(tmp_path)
    config['loras'] = [{'name': 'default', 'weight': 0.5}]
    record = {'prompt': 'Castle; literal text', 'seed': 7, 'loras': [{'name': 'scene', 'weight': 0.8}]}
    command = build_command(config, record, tmp_path / 'image.png')
    prompt = command[command.index('--prompt') + 1]
    assert prompt == 'Castle; literal text<lora:scene:0.8>'
    assert config['loras'][0]['name'] == 'default'
