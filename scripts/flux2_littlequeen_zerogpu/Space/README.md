---
title: Little Queen FLUX.2 Klein Trainer
sdk: gradio
sdk_version: 6.3.0
python_version: 3.10.13
app_file: app.py
startup_duration_timeout: 1h
models:
- black-forest-labs/FLUX.2-klein-base-4B
tags:
- zerogpu
- flux2-klein
- lora
- training
---

# Little Queen FLUX.2 Klein Trainer

Private segmented LoRA training endpoint. Each GPU call restores the latest complete
training state from the model repository, advances a bounded number of optimizer steps,
and uploads a new resumable checkpoint before returning.

