#!/usr/bin/env python3
"""Patch the pinned official trainer to support bounded resumable segments."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one patch target, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> int:
    args = parse_args()
    text = args.source.read_text(encoding="utf-8")
    text = replace_once(text, "import shutil\n", "import shutil\nimport time\n")
    text = replace_once(
        text,
        "def main(args):\n",
        "def main(args):\n    trainer_started = time.monotonic()\n",
    )
    text = replace_once(
        text,
        """            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
""",
        """            if os.path.isdir(args.dataset_name):
                dataset = load_dataset(
                    "imagefolder",
                    data_dir=args.dataset_name,
                    cache_dir=args.cache_dir,
                )
            else:
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                )
""",
    )
    text = replace_once(
        text,
        """    parser.add_argument(\n        \"--checkpointing_steps\",\n""",
        """    parser.add_argument(\n        \"--stop_after_step\",\n        type=int,\n        default=None,\n        help=(\n            \"Stop cleanly after this global step while keeping max_train_steps fixed for scheduler \"\n            \"continuity. A resumable checkpoint is always saved at the stop boundary.\"\n        ),\n    )\n    parser.add_argument(\n        \"--checkpointing_steps\",\n""",
    )
    text = replace_once(
        text,
        """        self.generator = random.Random(seed) if seed is not None else random\n\n        # Group indices by bucket\n""",
        """        self.seed = seed\n        self.epoch = 0\n\n        # Group indices by bucket\n""",
    )
    text = replace_once(
        text,
        """    def __iter__(self):\n        batches = []\n""",
        """    def set_epoch(self, epoch: int):\n        self.epoch = epoch\n\n    def __iter__(self):\n        generator = random.Random(self.seed + self.epoch) if self.seed is not None else random\n        batches = []\n""",
    )
    text = replace_once(text, "self.generator.shuffle(shuffled_indices)", "generator.shuffle(shuffled_indices)")
    text = replace_once(text, "self.generator.shuffle(batches)", "generator.shuffle(batches)")
    text = replace_once(
        text,
        """    if precompute_latents:
        cache_batch_sampler = BucketBatchSampler(
""",
        """    if precompute_latents:
        latent_cache_started = time.monotonic()
        cache_batch_sampler = BucketBatchSampler(
""",
    )
    text = replace_once(
        text,
        """    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if args.cache_latents:
""",
        """    if precompute_latents:
        logger.info(f"TIMING phase=latent_cache seconds={time.monotonic() - latent_cache_started:.3f}")

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if args.cache_latents:
""",
    )
    text = replace_once(
        text,
        """            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
""",
        """            checkpoint_restore_started = time.monotonic()
            accelerator.load_state(os.path.join(args.output_dir, path))
            logger.info(
                f"TIMING phase=checkpoint_restore seconds={time.monotonic() - checkpoint_restore_started:.3f}"
            )
            global_step = int(path.split("-")[1])
""",
    )
    text = replace_once(
        text,
        """    else:\n        initial_global_step = 0\n\n    progress_bar = tqdm(\n""",
        """    else:\n        initial_global_step = 0\n\n    resume_step = initial_global_step % num_update_steps_per_epoch
    logger.info(f"TIMING phase=trainer_initialization seconds={time.monotonic() - trainer_started:.3f}")
    optimizer_training_started = time.monotonic()
    checkpoint_save_seconds = 0.0

    progress_bar = tqdm(\n""",
    )
    text = replace_once(
        text,
        """                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
""",
        """                        checkpoint_save_started = time.monotonic()
                        accelerator.save_state(save_path)
                        checkpoint_save_seconds += time.monotonic() - checkpoint_save_started
                        logger.info(f"Saved state to {save_path}")
""",
    )
    text = replace_once(
        text,
        """    for epoch in range(first_epoch, args.num_train_epochs):\n        transformer.train()\n\n        for batch in train_dataloader:\n""",
        """    for epoch in range(first_epoch, args.num_train_epochs):\n        transformer.train()\n        batch_sampler = getattr(train_dataloader, \"batch_sampler\", None)\n        if hasattr(batch_sampler, \"set_epoch\"):\n            batch_sampler.set_epoch(epoch)\n        active_dataloader = train_dataloader\n        if args.resume_from_checkpoint and epoch == first_epoch and resume_step:\n            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)\n\n        for batch in active_dataloader:\n""",
    )
    text = replace_once(
        text,
        "    # Save the lora layers\n",
        """    optimizer_training_seconds = time.monotonic() - optimizer_training_started - checkpoint_save_seconds
    logger.info(f"TIMING phase=optimizer_training seconds={optimizer_training_seconds:.3f}")
    logger.info(f"TIMING phase=checkpoint_save seconds={checkpoint_save_seconds:.3f}")

    # Save the lora layers
""",
    )
    text = replace_once(
        text,
        "    accelerator.end_training()\n",
        """    logger.info(f"TIMING phase=trainer_total seconds={time.monotonic() - trainer_started:.3f}")
    accelerator.end_training()
""",
    )
    text = replace_once(
        text,
        """            if global_step >= args.max_train_steps:\n                break\n\n        if accelerator.is_main_process:\n""",
        """            stop_step = args.stop_after_step or args.max_train_steps\n            if global_step >= stop_step:\n                if accelerator.is_main_process or is_fsdp:\n                    save_path = os.path.join(args.output_dir, f\"checkpoint-{global_step}\")\n                    if not os.path.isdir(save_path):\n                        accelerator.save_state(save_path)\n                        logger.info(f\"Saved segment boundary state to {save_path}\")\n                break\n\n        if global_step >= (args.stop_after_step or args.max_train_steps):\n            break\n\n        if accelerator.is_main_process:\n""",
    )
    args.destination.write_text(text, encoding="utf-8")
    print(args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
