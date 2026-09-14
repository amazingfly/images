---
title: Little Queen LoRA Evaluator
sdk: gradio
sdk_version: 6.3.0
python_version: 3.10.13
app_file: app.py
startup_duration_timeout: 1h
models:
- black-forest-labs/FLUX.2-klein-base-4B
- amazingfly/little-queen-flux2-klein-lora-v1
- amazingfly/little-queen-flux2-klein-lora-v2
tags:
- zerogpu
- flux2-klein
- lora
- evaluation
---

# Little Queen LoRA Evaluator

Private, deterministic ZeroGPU endpoint for matched Little Queen LoRA evaluation.
Each request renders one locked job and atomically commits its PNG and provenance
metadata to a private results repository. Duplicate request IDs reuse the durable
Hub result instead of spending GPU quota again.
