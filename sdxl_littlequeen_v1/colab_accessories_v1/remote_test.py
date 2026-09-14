#!/usr/bin/env python3
"""Test the trained accessory LoRAs with the identity and outfit stack on Colab."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/lqaccessories_v1")
OUTPUT = ROOT / "test_output"
PROMPTS = Path("/content/lqaccessories_test_prompts.json")
ARCHIVE = Path("/content/lqaccessories_test_results.tar.gz")
MODEL = ROOT / "models" / "sd_xl_base_1.0.safetensors"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_FILE = "sd_xl_base_1.0.safetensors"
MODEL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
IDENTITY = ROOT / "test_models" / "lqxl_sdxl_v2.safetensors"
OUTFIT = ROOT / "test_models" / "lqmoonfit_sdxl_v1.safetensors"
REGALIA = ROOT / "output_regalia" / "lqmoonregalia_sdxl_v1.safetensors"
EQUIPMENT = ROOT / "output_equipment" / "lqhandgear_sdxl_v1.safetensors"
REGALIA_300 = ROOT / "test_models" / "lqmoonregalia_step300.safetensors"
EQUIPMENT_300 = ROOT / "test_models" / "lqhandgear_step300.safetensors"
EQUIPMENT_600 = ROOT / "test_models" / "lqhandgear_step600.safetensors"
WIDTH = 832
HEIGHT = 1216
STEPS = 28
GUIDANCE = 5.5
TEST_NAME = "littlequeen_accessories_v1_colab_consistency_test"

UPLOADS = (
    (
        Path("/content/lqxl_sdxl_v2.safetensors.part_"),
        IDENTITY,
        170_552_852,
        "3799128d4bfd4fbc7848d5b1de099a3e99cf3f718845c98efa7d4d48a7953994",
    ),
    (
        Path("/content/lqmoonfit_sdxl_v1.safetensors.part_"),
        OUTFIT,
        170_545_956,
        "985e82df407690831ef8d27a9c6c3b061689278c334e3694e21eaf8aa88c36d7",
    ),
    (
        Path("/content/lqmoonregalia_sdxl_v1.safetensors.part_"),
        REGALIA,
        85_425_732,
        "e5d606992392dafef990dcfb95ae7097289541365468def317405254d7b3acb4",
    ),
    (
        Path("/content/lqhandgear_sdxl_v1.safetensors.part_"),
        EQUIPMENT,
        170_544_084,
        "08eb0770724e2c7b165749185facd29da3c2163af7f5acde1529366ced35bf1e",
    ),
)

OPTIONAL_UPLOADS = (
    (
        Path("/content/lqmoonregalia_step300.safetensors.part_"),
        REGALIA_300,
        85_425_740,
        "482429d78678c0f54856a4e76b6cc63b05b685f628ff63097be7cd8ad8b566df",
    ),
    (
        Path("/content/lqhandgear_step300.safetensors.part_"),
        EQUIPMENT_300,
        170_544_084,
        "8c9a96909fbcaa02f079ebb5fe15e82ca239b3460060e71f04b60096f92a4a2f",
    ),
    (
        Path("/content/lqhandgear_step600.safetensors.part_"),
        EQUIPMENT_600,
        170_544_084,
        "d7de3d7a29220b1a4252d7d3510a4386b0d5155f1dd86da6aa91f7861e78ee85",
    ),
)

NEGATIVE = (
    "cropped head, cropped crown, cropped feet, out of frame, two girls, multiple "
    "people, duplicate, twin, adult woman, mature body, extra limbs, extra hands, "
    "missing limbs, malformed hands, asymmetrical crown, tilted broken crown, "
    "multiple crowns, floating crown, missing center gem, mismatched earrings, "
    "multiple weapons, duplicate wand, extra wand, bent shaft, broken wand, weapon "
    "through body, fused equipment, photorealistic, text, watermark, blurry, low detail"
)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(prefix: Path, destination: Path, expected_bytes: int, expected_sha: str) -> None:
    if destination.is_file() and destination.stat().st_size == expected_bytes and sha256(destination) == expected_sha:
        return
    parts = sorted(prefix.parent.glob(prefix.name + "*"))
    if not parts:
        raise FileNotFoundError(f"no chunks match {prefix}*")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if temporary.stat().st_size != expected_bytes or sha256(temporary) != expected_sha:
        raise RuntimeError(f"uploaded model validation failed: {destination.name}")
    temporary.replace(destination)


def valid_png(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return path.stat().st_size > 1024
    except (OSError, ValueError):
        return False


def ensure_model() -> None:
    if MODEL.is_file() and sha256(MODEL) == MODEL_SHA256:
        return
    from huggingface_hub import hf_hub_download

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILE, local_dir=MODEL.parent)
    )
    if downloaded != MODEL:
        shutil.copy2(downloaded, MODEL)
    actual = sha256(MODEL)
    if actual != MODEL_SHA256:
        raise RuntimeError(f"SDXL checksum mismatch: {actual}")


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
            "compel==2.0.3",
            "safetensors>=0.4.5",
        ]
    )
    for prefix, destination, expected_bytes, expected_sha in UPLOADS:
        assemble(prefix, destination, expected_bytes, expected_sha)
    for prefix, destination, expected_bytes, expected_sha in OPTIONAL_UPLOADS:
        if destination.is_file() or any(prefix.parent.glob(prefix.name + "*")):
            assemble(prefix, destination, expected_bytes, expected_sha)
    ensure_model()
    required = (MODEL, IDENTITY, OUTFIT, REGALIA, EQUIPMENT, PROMPTS)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    from compel import Compel, ReturnedEmbeddingsType
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    run(["nvidia-smi"])
    records = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading SDXL from verified local checkpoint {MODEL}", flush=True)
    pipe = StableDiffusionXLPipeline.from_single_file(
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
    pipe.load_lora_weights(str(IDENTITY), adapter_name="identity")
    pipe.load_lora_weights(str(OUTFIT), adapter_name="outfit")
    pipe.load_lora_weights(str(REGALIA), adapter_name="regalia")
    pipe.load_lora_weights(str(EQUIPMENT), adapter_name="equipment")
    optional_adapters = (
        (REGALIA_300, "regalia300"),
        (EQUIPMENT_300, "equipment300"),
        (EQUIPMENT_600, "equipment600"),
    )
    for path, adapter_name in optional_adapters:
        if path.is_file():
            pipe.load_lora_weights(str(path), adapter_name=adapter_name)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )
    negative, negative_pooled = compel(NEGATIVE)
    summary = {
        "name": TEST_NAME,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL),
        "resolution": [WIDTH, HEIGHT],
        "steps": STEPS,
        "guidance_scale": GUIDANCE,
        "default_weights": {"identity": 0.9, "outfit": 0.7},
        "results": [],
    }

    for index, record in enumerate(records, start=1):
        destination = OUTPUT / f"image_{index:02d}_{record['case_id']}_seed_{record['seed']}.png"
        # PEFT toggles adapter parameter flags while changing weights. The
        # offloaded SDXL modules are inference tensors, so keep that operation
        # in inference mode as well as the denoising call.
        with torch.inference_mode():
            regalia_adapter = record.get("regalia_adapter", "regalia")
            equipment_adapter = record.get("equipment_adapter", "equipment")
            identity_weight = float(record.get("identity_weight", 0.9))
            outfit_weight = float(record.get("outfit_weight", 0.7))
            adapter_weights = {
                "identity": identity_weight,
                "outfit": outfit_weight,
                regalia_adapter: float(record["regalia_weight"]),
                equipment_adapter: float(record["equipment_weight"]),
            }
            adapter_names = ["identity", "outfit", "regalia", "equipment"]
            adapter_names.extend(
                adapter_name
                for path, adapter_name in optional_adapters
                if path.is_file()
            )
            pipe.set_adapters(
                adapter_names,
                adapter_weights=[adapter_weights.get(name, 0.0) for name in adapter_names],
            )
        positive, pooled = compel(record["prompt"])
        positive, padded_negative = compel.pad_conditioning_tensors_to_same_length([positive, negative])
        started = time.monotonic()
        print(
            f"[{index:02d}/{len(records):02d}] {record['case_id']} "
            f"regalia={record['regalia_weight']} equipment={record['equipment_weight']}",
            flush=True,
        )
        with torch.inference_mode():
            image = pipe(
                prompt_embeds=positive,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=padded_negative,
                negative_pooled_prompt_embeds=negative_pooled,
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=torch.Generator(device="cuda").manual_seed(int(record["seed"])),
            ).images[0]
        image.save(destination, format="PNG", compress_level=6)
        if not valid_png(destination):
            raise RuntimeError(f"invalid output: {destination}")
        result = {
            **record,
            "index": index,
            "identity_weight": identity_weight,
            "outfit_weight": outfit_weight,
            "output": str(destination),
            "duration_seconds": round(time.monotonic() - started, 2),
            "status": "completed",
        }
        summary["results"].append(result)
        (ROOT / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        del image, positive, pooled, padded_negative
        torch.cuda.empty_cache()

    summary["status"] = "complete"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    (ROOT / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="output")
        archive.add(ROOT / "test_summary.json", arcname="test_summary.json")
    print(f"Wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
