#!/usr/bin/env python3
"""Validate V4 LoRAs as localized refiners of canonical accessory structure."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/lqaccessories_v4_final_region")
BUNDLE_PREFIX = Path("/content/lqaccessories_v4_final_region_bundle.tar.gz.part_")
REGALIA_PREFIX = Path("/content/lqmoonregalia_sdxl_v4.safetensors.part_")
WAND_PREFIX = Path("/content/lqmoonwand_sdxl_v4.safetensors.part_")
REGALIA_LORA = ROOT / "models" / "lqmoonregalia_sdxl_v4.safetensors"
WAND_LORA = ROOT / "models" / "lqmoonwand_sdxl_v4.safetensors"
MODEL = ROOT / "models" / "sd_xl_base_1.0.safetensors"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
ARCHIVE = Path("/content/lqaccessories_v4_final_region_results.tar.gz")
STRENGTHS = (0.12, 0.20, 0.28)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(prefix: Path, destination: Path) -> None:
    parts = sorted(prefix.parent.glob(prefix.name + "*"))
    if not parts:
        raise FileNotFoundError(f"no chunks match {prefix}*")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)


def ensure_model() -> None:
    from huggingface_hub import hf_hub_download

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="sd_xl_base_1.0.safetensors",
            local_dir=MODEL.parent,
        )
    )
    if downloaded != MODEL:
        shutil.copy2(downloaded, MODEL)
    if sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("SDXL checksum mismatch")


def make_pipeline(lora: Path, adapter: str, weight: float):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLInpaintPipeline

    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        str(MODEL),
        config=MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.load_lora_weights(str(lora), adapter_name=adapter)
    pipe.set_adapters([adapter], adapter_weights=[weight])
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def contact_sheet(records: list[dict], destination: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    columns = 3
    cell = (280, 400)
    sheet = Image.new("RGB", (columns * cell[0], len(records) * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        for column, key in enumerate(("input", "regalia", "final")):
            image = Image.open(record[key]).convert("RGB")
            thumb = ImageOps.contain(image, (cell[0] - 8, cell[1] - 28))
            x = column * cell[0] + (cell[0] - thumb.width) // 2
            y = row * cell[1]
            sheet.paste(thumb, (x, y))
            draw.text(
                (column * cell[0] + 4, y + cell[1] - 22),
                f"{record['id']} s={record['strength']:.2f} {key}",
                fill="white",
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=93)


def main() -> int:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "diffusers==0.34.0",
            "transformers==4.49.0",
            "accelerate==1.4.0",
            "peft==0.15.2",
            "safetensors>=0.4.5",
        ]
    )
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    run(["nvidia-smi"])
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    bundle = ROOT / "bundle.tar.gz"
    assemble(BUNDLE_PREFIX, bundle)
    assemble(REGALIA_PREFIX, REGALIA_LORA)
    assemble(WAND_PREFIX, WAND_LORA)
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(ROOT)
    ensure_model()

    manifest = json.loads((ROOT / "regional_selection.json").read_text(encoding="utf-8"))
    regalia_dir = ROOT / "regalia"
    final_dir = ROOT / "final"
    review_dir = ROOT / "review"
    for path in (regalia_dir, final_dir, review_dir):
        path.mkdir(parents=True, exist_ok=True)

    regalia_prompt = (
        "lqmoonregalia4, preserve the same exact five-peak polished gold Moonstar crown, "
        "one centered oval opalescent rose-amber moonstone, two smaller amber side gems, "
        "matched small gold leaf-drop earrings and narrow gold moonstone collar, integrated "
        "storybook anime linework and lighting"
    )
    regalia_negative = (
        "different crown, giant crown, extra crown peaks, flower crown, roses, hat, archway, "
        "cage, gold clothing, extra jewelry, duplicate, changed face, changed hair, changed dress"
    )
    records = []
    pipe = make_pipeline(REGALIA_LORA, "regalia", 0.45)
    for source in manifest["records"]:
        item_id = source["id"]
        image = Image.open(ROOT / "inputs" / f"{item_id}.png").convert("RGB")
        mask = Image.open(ROOT / "masks" / f"{item_id}_regalia.png").convert("L")
        for strength in STRENGTHS:
            tag = f"{item_id}_s{round(strength * 100):02d}"
            print(f"[regalia] {tag}", flush=True)
            with torch.inference_mode():
                result = pipe(
                    prompt=regalia_prompt,
                    negative_prompt=regalia_negative,
                    image=image,
                    mask_image=mask,
                    width=image.width,
                    height=image.height,
                    strength=strength,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                    padding_mask_crop=32,
                    generator=torch.Generator(device="cuda").manual_seed(
                        int(source["seed"]) + round(strength * 100)
                    ),
                ).images[0]
            destination = regalia_dir / f"{tag}.png"
            result.save(destination, format="PNG", compress_level=6)
            records.append(
                {
                    "id": item_id,
                    "strength": strength,
                    "seed": source["seed"],
                    "input": str(ROOT / "inputs" / f"{item_id}.png"),
                    "regalia": str(destination),
                }
            )
            del result
            torch.cuda.empty_cache()
    del pipe
    torch.cuda.empty_cache()

    wand_prompt = (
        "lqmoonwand4, preserve the same exact single slender gold Moonstar scepter, straight "
        "engraved shaft with moon and star motifs, broad open crescent finial around one large "
        "rose-amber oval moonstone, five small leaf crystals, two round side gems, one pointed "
        "rose crystal end cap, integrated storybook anime linework and lighting"
    )
    wand_negative = (
        "different wand, giant staff, sword, blade, star wand, flower, vine, archway, cage, "
        "freestanding object, extra wand, duplicate, bent shaft, changed hand, changed body"
    )
    pipe = make_pipeline(WAND_LORA, "wand", 0.50)
    for record in records:
        item_id = record["id"]
        image = Image.open(record["regalia"]).convert("RGB")
        mask = Image.open(ROOT / "masks" / f"{item_id}_wand.png").convert("L")
        strength = float(record["strength"])
        tag = f"{item_id}_s{round(strength * 100):02d}"
        print(f"[wand] {tag}", flush=True)
        with torch.inference_mode():
            result = pipe(
                prompt=wand_prompt,
                negative_prompt=wand_negative,
                image=image,
                mask_image=mask,
                width=image.width,
                height=image.height,
                strength=strength,
                num_inference_steps=28,
                guidance_scale=3.5,
                padding_mask_crop=32,
                generator=torch.Generator(device="cuda").manual_seed(
                    int(record["seed"]) + 500 + round(strength * 100)
                ),
            ).images[0]
        destination = final_dir / f"{tag}.png"
        result.save(destination, format="PNG", compress_level=6)
        record["final"] = str(destination)
        del result
        torch.cuda.empty_cache()

    contact_sheet(records, review_dir / "v4_final_regional_strength_sweep.jpg")
    summary = {
        "name": "littlequeen_accessories_v4_final_regional_validation",
        "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "strengths": list(STRENGTHS),
        "regalia_adapter_weight": 0.45,
        "wand_adapter_weight": 0.50,
        "base_model_sha256": sha256(MODEL),
        "regalia_lora_sha256": sha256(REGALIA_LORA),
        "wand_lora_sha256": sha256(WAND_LORA),
        "records": records,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in (regalia_dir, final_dir, review_dir):
            archive.add(path, arcname=path.name)
        archive.add(ROOT / "summary.json", arcname="summary.json")
    print(f"Wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
