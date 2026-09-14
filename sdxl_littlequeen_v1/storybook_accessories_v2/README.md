# Little Queen Storybook Accessories V2

V2 keeps the deterministic Moonstar crown and jewelry but replaces the
floor-length Rosekeeper staff with a compact hand-held Moonstar wand. It also
uses smaller wearable geometry and emits precise full-accessory masks for
localized SDXL/LoRA training and inpainting.

```bash
python scripts/apply_storybook_accessories.py INPUT.png OUTPUT.png \
  --set storybook_accessories_v2/accessory_set.json --staff
```

The output includes `_regalia_mask.png` and `_hand_prop_mask.png`. These masks
contain only visible accessory pixels plus a narrow contact edge; they should be
used for alpha-masked LoRA loss or localized inpainting. Broad face/torso boxes
must not be used for accessory training.
