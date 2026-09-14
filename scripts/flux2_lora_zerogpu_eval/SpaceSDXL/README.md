---
title: Little Queen SDXL Identity Evaluator
sdk: gradio
sdk_version: 6.3.0
python_version: 3.10.13
app_file: app.py
startup_duration_timeout: 1h
models:
- stabilityai/stable-diffusion-xl-base-1.0
- amazingfly/little-queen-sdxl-lora-v2
tags:
- zerogpu
- sdxl
- lora
- evaluation
---

# Little Queen SDXL Identity Evaluator

Private deterministic ZeroGPU endpoint for the second-stage Little Queen identity
comparison. Each result is committed atomically to the private evaluation dataset.
