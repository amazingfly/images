# Little Queen FLUX.2 Klein LoRA Training

This package rebuilds the Little Queen identity LoRA against
`black-forest-labs/FLUX.2-klein-base-4B` on Hugging Face ZeroGPU.

## Training design

- 36 manually reviewed images: 30 from the strongest SDXL identity dataset and six
  story images that add seated, rear, action, magic, outfit-color, and hair-style coverage.
- Native portrait aspect ratio, capped at 1024px.
- BF16 base model training, LoRA rank/alpha 32/32, learning rate `1e-4`.
- 1600 total optimizer steps. The primary segment is 215 steps with a dynamic
  200-second ZeroGPU declaration, based on measured fixed cost plus 0.8 seconds per step.
- After a primary segment, the supervisor calculates the largest one-step-increment
  tail whose admission reservation fits the quota reported by Hugging Face. Tail calls
  use an empirical `26 + 0.8 * steps` declaration based on measured warm-tail runtimes;
  this deliberately spends most of the conservative margin when the quota would
  otherwise be stranded.
- Full optimizer and scheduler state uploaded after every successful segment.
- At most three live resumable checkpoints plus inference-only LoRAs every 200 steps.
- The final adapter is published as `littlequeen-flux2-klein-v1.safetensors`.

The segmented trainer is based on Hugging Face Diffusers commit
`3a2f35d4efa4c059c8bfb3bc0d6c906264895c81`. A small deterministic patch adds a
bounded stop step and correct mid-epoch resume behavior while preserving the official
trainer's optimizer, scheduler, LoRA serialization, and dataset implementation.

## Durability

The local supervisor calls a private Gradio ZeroGPU Space. A successful call always:

1. restores the latest Hub checkpoint;
2. advances no more than one segment;
3. saves complete optimizer, scheduler, scaler, and random-number state;
4. uploads the checkpoint and `training_state.json` in one atomic Hub commit;
5. returns only after the Hub commit succeeds.

Request IDs make response-loss retries idempotent. On an explicit quota error, the
supervisor parses Hugging Face's exact retry countdown and atomically persists an
absolute UTC deadline in `supervisor_state.json`. Service or machine restarts continue
the same wait instead of starting a new 24-hour delay. If the response lacks a
countdown, it falls back to 24 hours. Other transient failures get five short retries
and then one retry per hour. It never rotates accounts.

Each segment result contains monotonic phase timings for Hub snapshots and uploads,
the GPU function and trainer process, initialization, latent caching, checkpoint
restore/save, optimizer training, and total/non-training endpoint time. Results are
also appended locally to `segment_metrics.jsonl` for direct comparison across days.

## Authentication

The configured token file is `/home/derek/projects/hug/token`. Deployment requires a
write token belonging to `amazingfly`; the currently discovered token is read-only.
The launcher waits without modifying or printing the token until that file contains a
write-capable token.

## Files

- `prepare_dataset.py`: deterministic dataset builder and provenance manifest.
- `deploy.py`: private dataset/model/Space deployment and ZeroGPU selection.
- `Space/app.py`: Gradio endpoint and checkpoint upload logic.
- `patch_trainer.py`: deterministic patch for the pinned official trainer.
- `supervise.py`: multi-day quota-aware controller.
- `start.sh`: prepare, deploy, and supervise entry point.
- `status.py`: read-only Hub status report.

## Start

Run it as a user service so the local controller survives terminal closure:

```bash
mkdir -p ~/.config/systemd/user
ln -sf \
  /home/derek/projects/agentic/ltxVideo/scripts/flux2_littlequeen_zerogpu/littlequeen-flux2.service \
  ~/.config/systemd/user/littlequeen-flux2.service
systemctl --user daemon-reload
systemctl --user enable --now littlequeen-flux2.service
```

The service stops normally when training reaches its target and restarts after
unexpected local failures.
