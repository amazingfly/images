#!/usr/bin/env python3
"""Structural tests for the ZeroGPU LoRA evaluator."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((HERE / CONFIG["manifest"]).read_text(encoding="utf-8"))
SDXL_CONFIG = json.loads((HERE / "config_sdxl.json").read_text(encoding="utf-8"))
SDXL_MANIFEST = json.loads(
    (HERE / SDXL_CONFIG["manifest"]).read_text(encoding="utf-8")
)


class EvaluationPackageTests(unittest.TestCase):
    def test_manifest_is_exactly_paired(self) -> None:
        jobs = MANIFEST["jobs"]
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job["id"] for job in jobs}), 12)
        grouped = {}
        for job in jobs:
            grouped.setdefault((job["scene"], job["lora_scale"]), []).append(job)
        self.assertEqual(set(grouped), {(2, 0.8), (2, 1.0), (88, 0.8), (88, 1.0), (92, 0.8), (92, 1.0)})
        for pair in grouped.values():
            self.assertEqual({job["variant"] for job in pair}, {"flux_v1", "flux_v2"})
            self.assertEqual(len({job["seed"] for job in pair}), 1)
            self.assertEqual(len({job["prompt"] for job in pair}), 1)

    def test_config_uses_completed_v1_space_without_touching_model_repos(self) -> None:
        self.assertEqual(
            CONFIG["space_repo"], "amazingfly/little-queen-flux2-klein-trainer"
        )
        self.assertNotEqual(CONFIG["result_repo"], CONFIG["adapters"]["flux_v1"]["repo"])
        self.assertNotEqual(CONFIG["result_repo"], CONFIG["adapters"]["flux_v2"]["repo"])
        self.assertEqual(CONFIG["gpu_duration_seconds"], 60)

    def test_space_has_one_gpu_entry_and_root_bf16_pipeline(self) -> None:
        source = (HERE / "Space/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        decorated = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any("spaces.GPU" in ast.unparse(item) for item in node.decorator_list):
                    decorated.append(node.name)
        self.assertEqual(decorated, ["render_gpu"])
        self.assertIn("torch_dtype=torch.bfloat16", source)
        self.assertIn('adapter_name="flux_v1"', source)
        self.assertIn('adapter_name="flux_v2"', source)
        self.assertIn('PIPE.to("cuda")', source)

    def test_results_are_idempotent_and_atomically_committed(self) -> None:
        source = (HERE / "Space/app.py").read_text(encoding="utf-8")
        self.assertIn("prior = existing_result(api, job)", source)
        self.assertIn("api.create_commit(", source)
        self.assertIn("CommitOperationAdd(path_in_repo=image_remote", source)
        self.assertIn("path_in_repo=metadata_remote", source)
        self.assertIn('"image_sha256": record["image"]["sha256"]', source)
        self.assertNotIn('return {"status": "completed", **record}', source)

    def test_quota_parser_and_sheet(self) -> None:
        import supervise

        error = RuntimeError("Try again in 23:47:52. Subscribe to Hugging Face PRO")
        self.assertEqual(supervise.quota_retry_after_seconds(error), 85672)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            jobs = MANIFEST["jobs"][:2]
            for job, color in zip(jobs, ("red", "blue")):
                image, metadata = supervise.local_paths(output, job["id"])
                image.parent.mkdir(parents=True, exist_ok=True)
                metadata.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (832, 1216), color).save(image)
                metadata.write_text("{}", encoding="utf-8")
            sheet = supervise.build_sheet(output, jobs)
            with Image.open(sheet) as image:
                self.assertEqual(image.size, (832, 654))

    def test_sdxl_manifest_matches_promoted_flux_jobs(self) -> None:
        sdxl_jobs = SDXL_MANIFEST["jobs"]
        self.assertEqual(len(sdxl_jobs), 3)
        self.assertEqual({job["scene"] for job in sdxl_jobs}, {2, 88, 92})
        promoted = {
            job["scene"]: job
            for job in MANIFEST["jobs"]
            if job["variant"] == "flux_v2" and job["lora_scale"] == 1.0
        }
        for job in sdxl_jobs:
            baseline = promoted[job["scene"]]
            self.assertEqual(job["seed"], baseline["seed"])
            self.assertEqual(job["prompt"], baseline["prompt"])
            self.assertEqual(job["lora_scale"], 1.0)
            self.assertEqual(job["variant"], "sdxl_v2")

    def test_sdxl_config_preserves_reviewed_identity_recipe(self) -> None:
        self.assertEqual(SDXL_CONFIG["width"], 768)
        self.assertEqual(SDXL_CONFIG["height"], 1024)
        self.assertEqual(SDXL_CONFIG["num_inference_steps"], 28)
        self.assertEqual(SDXL_CONFIG["guidance_scale"], 5.0)
        self.assertEqual(SDXL_CONFIG["trigger"], "lqxl Little Queen")
        self.assertEqual(
            SDXL_CONFIG["adapter"]["sha256"],
            "3799128d4bfd4fbc7848d5b1de099a3e99cf3f718845c98efa7d4d48a7953994",
        )
        self.assertTrue(Path(SDXL_CONFIG["adapter"]["source"]).is_file())

    def test_sdxl_space_has_exact_fp16_recipe_and_atomic_results(self) -> None:
        source = (HERE / "SpaceSDXL/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        decorated = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any("spaces.GPU" in ast.unparse(item) for item in node.decorator_list):
                    decorated.append(node.name)
        self.assertEqual(decorated, ["render_gpu"])
        self.assertIn("torch_dtype=torch.float16", source)
        self.assertIn('variant="fp16"', source)
        self.assertIn('algorithm_type="dpmsolver++"', source)
        self.assertIn("use_karras_sigmas=True", source)
        self.assertIn('PIPE.to("cuda")', source)
        self.assertNotIn("LCMScheduler", source)
        self.assertIn("api.create_commit(", source)
        self.assertIn('"image_sha256": record["image"]["sha256"]', source)

    def test_cross_model_sheet(self) -> None:
        import supervise

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            baseline = [
                job
                for job in MANIFEST["jobs"]
                if job["variant"] == "flux_v2" and job["lora_scale"] == 1.0
            ]
            for job in [*baseline, *SDXL_MANIFEST["jobs"]]:
                image, metadata = supervise.local_paths(output, job["id"])
                image.parent.mkdir(parents=True, exist_ok=True)
                metadata.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (768, 1024), "gold").save(image)
                metadata.write_text("{}", encoding="utf-8")
            sheet = supervise.build_cross_model_sheet(
                output, SDXL_MANIFEST["jobs"], baseline
            )
            with Image.open(sheet) as image:
                self.assertEqual(image.size, (832, 1962))


if __name__ == "__main__":
    unittest.main()
