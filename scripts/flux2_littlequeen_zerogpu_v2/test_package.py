#!/usr/bin/env python3
"""Fast structural tests for the segmented FLUX.2 training package."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from datasets import load_dataset
from PIL import Image


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


class PackageTests(unittest.TestCase):
    def test_dataset_integrity(self) -> None:
        dataset = Path(CONFIG["dataset_dir"])
        manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["image_count"], 30)
        self.assertEqual(len(manifest["images"]), 30)
        self.assertEqual(len({item["sha256"] for item in manifest["images"]}), 30)
        self.assertEqual(manifest["excluded_source_indices"], [8, 18, 20, 31, 32, 34])
        for item in manifest["images"]:
            image_path = dataset / item["image"]
            caption_path = dataset / item["caption_file"]
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"])
            caption = caption_path.read_text(encoding="utf-8")
            self.assertTrue(caption.startswith("LQK4N. "))
            for leaked_identity_word in ("hair", "bangs", "eyes", "face", "auburn"):
                self.assertNotIn(leaked_identity_word, caption.lower())
            with Image.open(image_path) as image:
                image.verify()

    def test_dataset_loads_as_imagefolder(self) -> None:
        dataset = load_dataset("imagefolder", data_dir=CONFIG["dataset_dir"], split="train")
        self.assertEqual(dataset.column_names, ["image", "text"])
        self.assertEqual(dataset.num_rows, 30)
        self.assertEqual(dataset[0]["image"].size, (832, 1216))
        self.assertTrue(dataset[0]["text"].startswith("LQK4N. "))

    def test_config_is_consistent(self) -> None:
        self.assertEqual(CONFIG["base_model"], "black-forest-labs/FLUX.2-klein-base-4B")
        self.assertEqual(CONFIG["trigger"], "LQK4N")
        self.assertEqual(CONFIG["total_steps"], 1200)
        self.assertTrue(CONFIG["space_repo"].endswith("-v2"))
        self.assertTrue(CONFIG["dataset_repo"].endswith("-v2"))
        self.assertTrue(CONFIG["output_repo"].endswith("-v2"))
        self.assertEqual(CONFIG["segment_steps"], 215)
        self.assertEqual(CONFIG["tail_step_increment"], 1)
        self.assertEqual(CONFIG["tail_min_steps"], 1)
        self.assertEqual(CONFIG["tail_gpu_duration_fixed_seconds"], 26)
        self.assertEqual(CONFIG["tail_gpu_seconds_per_step"], 0.8)
        self.assertEqual(
            min(
                CONFIG["gpu_max_duration_seconds"],
                math.ceil(
                    CONFIG["gpu_duration_fixed_seconds"]
                    + CONFIG["gpu_seconds_per_step"] * CONFIG["segment_steps"]
                ),
            ),
            CONFIG["gpu_max_duration_seconds"],
        )
        self.assertEqual(CONFIG["quota_initial_sleep_seconds"], 24 * 60 * 60)
        self.assertEqual(CONFIG["quota_retry_seconds"], 15 * 60)

    def test_space_exposes_only_one_gpu_entry(self) -> None:
        tree = ast.parse((HERE / "Space/app.py").read_text(encoding="utf-8"))
        decorated = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if "spaces.GPU" in ast.unparse(decorator):
                        decorated.append(node.name)
        self.assertEqual(decorated, ["train_gpu_segment"])

    def test_quota_tail_parser(self) -> None:
        import supervise

        message = RuntimeError(
            "You have exceeded your free ZeroGPU quota (135s requested vs. 111s left)."
        )
        self.assertEqual(supervise.quota_tail_steps(message, CONFIG), 60)
        self.assertEqual(
            supervise.quota_tail_steps(
                RuntimeError("135s requested vs. 68s left"), CONFIG
            ),
            23,
        )
        self.assertEqual(
            supervise.quota_tail_steps(
                RuntimeError("135s requested vs. 54s left"), CONFIG
            ),
            12,
        )
        self.assertEqual(
            supervise.quota_tail_steps(
                RuntimeError("135s requested vs. 53s left"), CONFIG
            ),
            11,
        )
        self.assertEqual(
            supervise.quota_tail_steps(
                RuntimeError("135s requested vs. 43s left"), CONFIG
            ),
            2,
        )
        self.assertEqual(
            supervise.quota_tail_steps(
                RuntimeError("135s requested vs. 40s left"), CONFIG
            ),
            0,
        )

    def test_quota_retry_parser(self) -> None:
        import supervise

        error = RuntimeError("Try again in 23:47:52. Subscribe to Hugging Face PRO")
        self.assertEqual(supervise.quota_retry_after_seconds(error), 85672)

    def test_supervisor_state_survives_restart(self) -> None:
        import supervise

        with tempfile.TemporaryDirectory() as temp:
            state_path = Path(temp) / "supervisor_state.json"
            state = {"version": "test"}
            error = RuntimeError(
                "135s requested vs. 68s left. Try again in 00:10:00."
            )
            supervise.set_quota_deadline(state_path, state, error, CONFIG)
            restored = supervise.load_supervisor_state(state_path)
            self.assertEqual(restored["quota_remaining_seconds"], 68)
            self.assertIsNotNone(restored["next_attempt_at"])
            supervise.clear_quota_deadline(state_path, restored)
            self.assertIsNone(
                supervise.load_supervisor_state(state_path)["next_attempt_at"]
            )

    def test_checkpoint_and_state_are_one_commit(self) -> None:
        source = (HERE / "Space/app.py").read_text(encoding="utf-8")
        self.assertIn(
            'primary_patterns = [f"{checkpoint.name}/**", "training_state.json"]', source
        )
        self.assertIn('primary_patterns.append("littlequeen-flux2-klein-v2.safetensors")', source)
        self.assertIn("allow_patterns=primary_patterns", source)
        self.assertNotIn('path_in_repo="training_state.json"', source)

    def test_pinned_trainer_patch(self) -> None:
        source = Path("/tmp/train_dreambooth_lora_flux2_klein.py")
        if not source.is_file():
            self.skipTest("official trainer source is not cached locally")
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "trainer.py"
            subprocess.run(
                ["python3", str(HERE / "patch_trainer.py"), str(source), str(destination)],
                check=True,
                capture_output=True,
                text=True,
            )
            compile(destination.read_text(encoding="utf-8"), str(destination), "exec")
            patched = destination.read_text(encoding="utf-8")
            self.assertIn("--stop_after_step", patched)
            self.assertIn('load_dataset(\n                    "imagefolder"', patched)
            self.assertIn("accelerator.skip_first_batches", patched)
            self.assertIn("Saved segment boundary state", patched)


if __name__ == "__main__":
    unittest.main()
