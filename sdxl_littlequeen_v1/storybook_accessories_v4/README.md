# Little Queen Storybook Accessories V4

V4 uses trained LoRAs for appearance and fixed transparent assets for accessory
structure. Global LoRA prompting alone is not the production method: it varied
crown shape, changed wand designs, and spread gold detail into clothing.

## Production Pipeline

Run:

```bash
./colab_storybook_accessories_v4_auto/run_check.sh
```

The unattended runner generates up to 52 identity/outfit candidates at
832x1216 and automatically:

1. Creates a source hand pose around a plain guide rod.
2. Detects the Little Queen, face, hands, headwear, and props.
3. Segments the hand and crown-cleanup areas.
4. Composites the exact five-peak Moonstar crown, matched earrings/collar, and
   crescent Moonstar wand.
5. Refines regalia at LoRA weight `0.45`, strength `0.28` (`0.45` only for
   crown cleanup), 28 DPM++ Karras steps, guidance `3.5`.
6. Refines the complete wand at LoRA weight `0.50`, strength `0.30`.
7. Redraws only the small hand/shaft contact region at strength `0.68`, with
   wand LoRA reduced to `0.20` so base SDXL can form the fingers.
8. Runs remote and local validation and emits enlarged grip crops.

Every regional operation is bounded by an emitted mask. Nonzero feather pixels
define the allowed edit area; correlation and edge-retention checks use the
mask interior.

## Mask Artifacts

For every candidate, the compositor emits:

- `*_regalia_mask.png`: crown, earrings, collar, and confirmed cleanup region
- `*_wand_visible_mask.png`: canonical wand pixels visible after hand occlusion
- `*_wand_full_mask.png`: complete canonical wand silhouette
- `*_hand_occlusion_mask.png`: source hand restored in front of the wand
- `*_wand_mask.png`: wand plus hand region used by regional refinement
- `*_hand_grip_mask.png`: tight rounded region used only for grip repair
- `*.json`: detections, placements, source boxes, model-independent geometry

## Validation Scope

The deterministic validator covers framing, face/crown scale, clipping,
selected-hand position, wand continuity, hand/wand overlap, fragmentation,
face overlap, likely duplicate props, inherited headwear, exact edit scope,
edge retention, correlation, and sharpness.

Subtle finger semantics remain the hard case. Experiments with Gemma, ViTPose,
zero-shot CLIP, and a small CLIP linear classifier were not precise enough to
use as hard gates. `review/grip_contacts.jpg` is generated automatically so the
remaining visual audit is fast and explicit rather than hidden behind a
misleading automatic score.

## Calibration Evidence

The validator was compared with full-resolution personal decisions, not only
its own scores:

- The original open-palm run had 3 human accepts among 9 automatic accepts
  (`33%` accept precision).
- Guide-rod generation produced 5 human accepts among 13 refined candidates,
  improving source-pose yield but leaving open-hand false accepts.
- The first grip micro-pass still produced only 5 human accepts among 12
  geometry-valid finals, which exposed the disconnected-mask design error.
- A leave-one-run-out CLIP linear experiment reached about `73%` agreement and
  `63%` accept precision at its useful threshold, so it remains disabled.

Human labels and confusion reports are stored beside each checked run under
`outputs/storybook_accessories_v4_auto/*/review/`.

## Main Artifacts

- Configuration: `auto_pipeline_config.json`
- Colab bundle: `colab/storybook_accessories_v4_auto_bundle.tar.gz`
- Regalia LoRA: `models/loras/lqmoonregalia_sdxl_v4.safetensors`
- Wand LoRA: `models/loras/lqmoonwand_sdxl_v4.safetensors`
- Compositor: `scripts/apply_storybook_accessories.py`
- Validator: `scripts/validate_storybook_accessories_v4.py`
- Calibration: `scripts/calibrate_storybook_accessory_validator.py`
