> Story orchestration and its tests have moved to [storybook-pipeline](https://github.com/amazingfly/storybook-pipeline). This directory owns the image backends, training, and assets. Historical story commands below describe the original layout; use the new repository for fresh installations.

# Little Queen SDXL CPU v1

This is a separate SDXL path. It does not modify or replace the existing SD1.5
Little Queen generator.

## Local inference design

- Official SDXL 1.0 base checkpoint retained at full precision for Colab training.
- Q8_0 GGUF base for local CPU inference with `stable-diffusion.cpp`.
- Official SDXL LCM LoRA for 4-8 step CPU generation.
- Six physical CPU threads and memory-mapped weights. Tiled VAE decoding remains
  available as a low-memory fallback but is disabled when untiled decoding passes
  the memory guard.
- One render process at a time with one-second RAM and swap telemetry.
- Automatic termination below 1.2 GiB available RAM, above 12.5 GiB process RSS,
  or above 4 GiB total swap use.

Run setup and quantization:

```bash
./setup_sdxl_cpu.sh
```

Run a single smoke image:

```bash
./run_test10.sh --limit 1
```

Run all ten baseline images:

```bash
./run_test10.sh
```

Each run writes images and a summary under `outputs/test10/<timestamp>/`. Detailed
process logs and one-second memory samples are written under
`logs/test10/<timestamp>/`.

## Curated SDXL LoRA dataset

The training set is a manual full-resolution selection of 30 images from the
authoritative accepted curation report. The source files remain untouched.

- `training_dataset/curated_manifest.json` contains the selection and hand-written
  captions.
- `training_dataset/final_selection_sheets/` contains the two labeled review
  sheets used for the final visual check.
- `training_dataset/colab/prepared_manifest.json` records the source path and
  SHA-256 of every selected image.
- Captions deliberately omit invariant identity and color traits. They describe
  hairstyle, dress construction, accessory type, pose, framing, and scene so
  those elements remain controllable instead of becoming inseparable from the
  `lqxl` identity token.

Rebuild the paired image/caption directory without altering the sources:

```bash
./scripts/prepare_training_dataset.py
```

## Colab training

The Colab path uses a pinned `kohya-ss/sd-scripts` revision and the official,
unmodified SDXL 1.0 checkpoint. It trains a U-Net-only rank-32 LoRA for 1,200
steps on a T4 with FP16, SDPA, gradient checkpointing, cached text outputs, and
cached latents. The trainer's Colab-specific low-RAM loader places checkpoint
components directly on the larger VRAM device instead of duplicating them in
host RAM. Latents and caption embeddings are precomputed in isolated VAE-only
and text-encoder-only subprocesses before the U-Net-only training process starts.
Host RAM and GPU memory are sampled every ten seconds.

```bash
./colab/run_training.sh
```

On success, the runner downloads the result to
`models/loras/lqxl_sdxl_v1.safetensors` and stops the Colab runtime. The download
stage refreshes Colab's rotating proxy token before transferring artifacts. On
failure, it preserves the session by default so a finished remote output can be
recovered with `scripts/recover_colab_artifact.py`.

## Local character test

After the LoRA has downloaded, generate the five character-consistency tests:

```bash
./run_lora5.sh
```

This stacks the character LoRA with the SDXL LCM acceleration LoRA and retains
the same one-process-at-a-time RAM guard used for the baseline test. Visual A/B
testing selected the 900-step checkpoint at weight 0.65 for production: it keeps
the face and child proportions while allowing materially better framing, dress,
and hairstyle variation than the overbound 1,200-step model at weight 0.85. The
raw 1,200-step result remains archived as
`models/loras/lqxl_sdxl_v1-final1200.safetensors`.

## Seed-refine comparison

A controlled ten-pair experiment compared the production 768x1024 eight-step
path against a native high-resolution-fix path: four steps at 512x704, Lanczos
resize to 768x1024, then three low-noise img2img steps at strength 0.35. Prompts,
seeds, model, LoRAs, sampler, and safety limits were identical.

| Result | Standard | Seed-refine |
| --- | ---: | ---: |
| Mean time per image | 750.5 s | 660.4 s |
| Total time for ten | 7,505.5 s | 6,604.5 s |
| Peak process RSS | 10,131.8 MiB | 10,207.4 MiB |
| Mean gradient RMS | 0.043823 | 0.034778 |
| Mean Laplacian variance | 0.009988 | 0.004306 |

Seed-refine was 12.0% faster, but visual review preferred standard in all ten
pairs under the face/proportions/anatomy priorities. The gentle img2img pass
preserved low-resolution anatomy and prompt errors, and results were visibly
softer. Standard remains the production default; seed-refine is retained as an
optional draft mode in `config_compare_seed_refine10.json`.

Re-run the full comparison (about four hours on this CPU) with:

```bash
./run_seed_refine_comparison.sh
```

The completed report, paired sheets, and manual assessment are under
`outputs/seed_refine_comparison/report_20260719/`.

## FP16 Colab dataset generation and v2 LoRA

The v2 workflow is isolated from the CPU/v1 path above. On a T4 it uses SDXL in
FP16; Q8 is only useful for the constrained local CPU path and would needlessly
reduce generation quality on the GPU. Dataset images are rendered at 832x1216,
which is approximately SDXL's native one-megapixel scale and fits an exact 1024
training bucket without aspect-ratio cropping. Final inference remains 768x1024
to match the intended local production workload.

Generate the 100-image identity dataset with the v1 LoRA from the configured
Google Drive file:

```bash
./colab_generation/run_generation.sh full
```

The Drive file ID and verified SHA-256 are pinned in
`colab_generation/remote_generate.py`. Outputs are downloaded under
`outputs/identity_dataset100_v2/colab_fp16_832x1216_20260719/`.

The 30 manually selected images and hand-written captions are recorded in
`training_dataset_v2/curated_manifest.json`. Rebuild the paired training folder,
then train the separate v2 LoRA with:

```bash
./scripts/prepare_training_dataset_v2.py
./colab_v2/run_training.sh
```

Training uses the official SDXL base, FP16, rank 32, exact 832x1216 buckets, and
1,200 U-Net steps. The selected final checkpoint is
`models/loras/lqxl_sdxl_v2.safetensors`; its hash, inference settings, and manual
review are recorded in `models/loras/lqxl_sdxl_v2.selection.json`.

Run the selected checkpoint's ten-image identity test on a T4 with:

```bash
./colab_v2_test/run_test.sh final 1200
```

The intentionally minimal positive prompt form is `solo, lqxl Little Queen,
full-body, standing in <scene>`. Because this is a U-Net-only LoRA, `lqxl` alone
is not a learned tokenizer embedding; retaining the semantic class phrase
`Little Queen` is required for reliable subject grounding.

Run the separate 100-image local Q8/LCM production set with:

```bash
./run_lora_v2_local100.sh --run-id <run-name>
```

The deterministic prompt builder combines 25 locations with four simple action
types. It retains only the semantic character name, composition, scene, and
style rather than repeating visual identity traits. A stopped run can continue
with the same run name and `--resume`.
