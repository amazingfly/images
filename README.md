# Image generation and storybook workflows

Stable Diffusion/SDXL image generation, Little Queen LoRA training, FLUX ZeroGPU
experiments, and storybook validation/rendering. This repository owns the image
stage used by `sa3`, `ltx-video`, and `media-pipeline`.

## Choose a workflow

| Workflow | Entry point | Notes |
| --- | --- | --- |
| SD 1.5 images | `run_pipeline.sh` | Prompt generation, image generation, upscaling |
| SD 1.5 LoRA | `run_pipeline_lora.sh` | Uses `config_lora_littlequeen.json` |
| Curated image dataset | `run_littlequeen_dataset.sh` | Configurable dataset generation and curation |
| SDXL generation/training | [SDXL guide](sdxl_littlequeen_v1/README.md) | Local CPU and versioned Colab workers |
| Storybook compilation/rendering | [storybook guide](sdxl_littlequeen_v1/storybook_mvp_v1/README.md) | Scripts, schemas, candidate selection and page rendering |
| Qwen visual review | [local Qwen guide](scripts/qwen35_storybook_local/README.md) | Validate candidates, narrate, assemble story videos |
| Qwen Colab review | [Colab Qwen guide](scripts/qwen35_storybook_validation/README.md) | Dataset upload and remote validation |
| FLUX training v3 | [FLUX v3 guide](flux2_littlequeen_zerogpu_v3/README.md) | Current versioned training configuration |
| FLUX v1/v2 and evaluation | `scripts/flux2_*` | Earlier training and evaluation packages |

Versioned workflow directories retain their names because run manifests and
external launchers refer to them. Each workflow's README/config is its own entry
point; GPU jobs, deployments, and test packages requiring local datasets are not
run by the CPU test suite.

## Local setup

For the SD 1.5 scripts, use Python 3.11/3.12 and a dedicated environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-sd15.txt
PATH="$PWD/.venv/bin:$PATH" ./run_pipeline.sh
```

Provide model weights locally and adjust model/prompt-server paths in the selected
config before rendering. The default config requests local-only model loading.
SDXL has its own `sdxl_littlequeen_v1/setup_sdxl_cpu.sh`; Colab/Space workflows
carry separate remote dependency/setup files. FFmpeg and a Piper voice are needed
for narrated story videos. These GPU and model setups are workflow-specific and
are not installed by the lightweight checks below.

## CPU checks

```bash
uv venv .venv-check
uv pip install --python .venv-check/bin/python -r requirements-dev.txt
.venv-check/bin/python -m pytest
```

Tests cover existing storybook selection, schema/compilation, approved accessory
exports, and Qwen response/validation behavior. They use temporary fixtures and do
not require model downloads or cloud access.

## Source and local data

- `scripts/`: SD 1.5, dataset tools, and migrated FLUX/Qwen workflows.
- `sdxl_littlequeen_v1/`: SDXL and storybook source, presets, and tests.
- `flux2_littlequeen_zerogpu_v3/`: versioned FLUX training source.
- `agy/`: alternative local storybook accessory workflow.
- `theWitchesTrick/`: story source examples.
- `archive/`: earlier SA3-bundled image workflow, preserved for reference.
- `LORA/`, model/dataset/output directories, logs, and runtime state: ignored local data.

The root `flux2_lora_zerogpu_eval/` is **runtime data**; the evaluation **source**
is `scripts/flux2_lora_zerogpu_eval/`. See [migration notes](docs/migration.md)
for original Git-history preservation and compatibility paths.

Gemma-based curation can use the LTX launchers through `LTX_REPO=/path/to/ltx-video`
or explicit `--gemma-start`/`--gemma-stop` arguments. The Qwen workflow has its
own launchers. Service files are workstation examples; adjust paths when installing
on another machine. Existing installed services were not restarted by migration.
