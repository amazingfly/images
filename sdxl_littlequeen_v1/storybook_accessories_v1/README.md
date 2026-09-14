# Little Queen Storybook Accessories V1

This set uses fixed alpha assets for page-to-page design consistency. The accessory
LoRAs remain available for style guidance, but they are not the source of truth for
prop geometry.

## Canonical set

- `assets/moonstar_crown_trimmed.png`: five-flame moonstone diadem
- `assets/moonstar_earring_trimmed.png`: one earring master, mirrored for a pair
- `assets/moonstar_necklace_trimmed.png`: moonstone collar necklace
- `assets/rosekeeper_staff_trimmed.png`: straight Rosekeeper wand-staff

The source chroma images are retained under `source/` for audit and future edits.

## Placement

```bash
python scripts/apply_storybook_accessories.py INPUT.png OUTPUT.png --staff
```

The compositor detects the face, body, hands, and any inherited crown. It writes
the composited page, placement metadata, a localized `_blend_mask.png`, and an
old-crown `_cleanup_mask.png`. The staff is placed behind the segmented queen so
her hand and body occlude its shaft. Crown and jewelry are placed above the
character layer. Existing crowns are not removed by default. Pass
`--remove-existing-crown` only for a crown with normal background headroom; local
color inpainting cannot reconstruct hidden hair or scenery when a crown touches
the frame edge.

For repeatable art direction, override automatic placement with normalized values:

```bash
python scripts/apply_storybook_accessories.py INPUT.png OUTPUT.png \
  --staff --staff-anchor 0.30 0.46 --staff-angle -6 \
  --face-box 0.41 0.21 0.59 0.35
```

The reliable production input is a bare-headed, empty-handed base page. The
Colab v2 workflow in `../colab_storybook_accessories_v2/` sweeps identity-LoRA
weights and rejects bases with inherited props before compositing. Use the blend
mask only for a low-strength attachment pass around the crown band, earring
studs, necklace edge, and staff grip. Do not inpaint the complete prop or SDXL
will reinterpret its design.
