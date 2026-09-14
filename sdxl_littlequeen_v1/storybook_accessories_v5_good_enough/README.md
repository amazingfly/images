# Little Queen Storybook Accessories V5 Good Enough

This version prioritizes reaching a usable storybook baseline without training
another LoRA. It leaves V4 intact and keeps the same canonical accessory assets
and LoRA weights.

## Changes From V4

1. Base prompts use static upright poses with one hand clearly visible at waist
   height around a vertical guide rod.
2. The candidate pool increases from 52 to 64.
3. Both the pre-grip `wand_only` image and the post-grip `grip_repaired` image
   are retained.
4. Full-page and enlarged grip comparisons are emitted side by side.
5. Automatic geometry passes form a review pool, not a misleading final
   selection.
6. Personally approved variants are materialized into `approved/`.

Run:

```bash
./colab_storybook_accessories_v5_good_enough/run_check.sh
```

Important files:

- `auto_pipeline_config.json`: conservative scenes and generation settings
- `colab/storybook_accessories_v5_good_enough_bundle.tar.gz`: generated bundle
- `scripts/finalize_storybook_accessories_v5.py`: local approval finalizer
- `scripts/validate_storybook_accessories_v4.py`: unchanged geometry validator
