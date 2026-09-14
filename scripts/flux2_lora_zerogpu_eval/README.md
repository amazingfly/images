# Little Queen ZeroGPU LoRA Evaluation

This package performs a durable matched comparison of the completed FLUX.2 Klein
Little Queen v1 and v2 identity LoRAs. Stage one repeats the exact prior six jobs
for both adapters with identical prompts, seeds, BF16 base pipeline, dimensions,
guidance, and 50-step denoising. Results are committed one job at a time to a
private Hub dataset and recovered locally under the configured output directory.

The evaluator reuses the completed v1 trainer Space because the account's two
ZeroGPU Space slots are already occupied. It does not modify either trained LoRA
repository. The supervisor respects Hugging Face quota countdowns, persists the
absolute retry time, resumes after restarts, and exits when every job is durable.

The first two completed scene pairs promote v2: it has more stable facial
proportions, hair, crown, clothing, and instruction-following than v1. The stage
two service still waits for all 12 stage-one results before changing the shared
Space. It then compares FLUX v2 at weight 1.0 against the reviewed final 1200-step
SDXL identity checkpoint on all three matched prompts. SDXL uses the exact
known-good FP16 recipe from `minimal10_1200`: SDXL 1.0, DPM-Solver++ with Karras
sigmas, 28 steps, CFG 5.0, 768x1024, and LoRA weight 1.0. It does not use the
later 8-step LCM shortcut.
