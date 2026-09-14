---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-base-4B
library_name: diffusers
pipeline_tag: text-to-image
tags:
- lora
- flux2-klein
- character-consistency
---

# Little Queen FLUX.2 Klein Identity LoRA v3

Durable segmented training repository for the Little Queen character identity.

- Base training model: `black-forest-labs/FLUX.2-klein-base-4B`
- Inference-compatible model: `black-forest-labs/FLUX.2-klein-4B`
- Trigger: `LQK4N`
- Target steps: 1400
- Learning rate: 7e-5
- Dataset: 20 selected SDXL-derived originals and 8 portrait crops
- LoRA rank/alpha: 32/32
- Training precision: BF16
- Resolution: native-aspect buckets capped at 1024px

The repository keeps full optimizer checkpoints during training and publishes
`littlequeen-flux2-klein-v3.safetensors` after the final segment.
