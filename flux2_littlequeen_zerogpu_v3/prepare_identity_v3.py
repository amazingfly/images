"""Build an auditable identity dataset from reviewed SDXL-derived originals."""
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IDENTITY = (
    "LQK4N, a little queen with a rounded childlike face, large warm brown eyes, "
    "a small nose, soft round cheeks, and long chestnut-brown hair with bangs"
)
# These are V2 indices, not the earlier source indices.
KEEP = [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 22, 25, 26, 27, 28]
CROPS = {
    1: (110, 40, 710, 640), 3: (140, 100, 740, 700),
    7: (100, 0, 700, 600), 9: (180, 0, 780, 600),
    10: (130, 0, 730, 600), 14: (80, 0, 680, 600),
    15: (100, 0, 700, 600), 22: (120, 0, 720, 600),
}


def main():
    config = json.loads((HERE / 'config.json').read_text())
    target = Path(config['dataset_dir'])
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'sdxl_littlequeen_v1/training_dataset_flux2_klein_v2'
    original = json.loads((source / 'dataset_manifest.json').read_text())
    records = []
    for rec in original['images']:
        index = rec['index']
        if index not in KEEP:
            continue
        src = source / rec['image']
        variants = [('full', None)]
        if index in CROPS:
            variants.append(('portrait', CROPS[index]))
        for variant, box in variants:
            name = f'lqk4n3_{index:03d}_{variant}.png'
            dst = target / name
            with Image.open(src) as im:
                if box:
                    im.crop(box).save(dst)
                else:
                    shutil.copy2(src, dst)
            if box:
                caption = IDENTITY + '. Solo head-and-shoulders portrait, wearing a golden crown and a pink dress, in a storybook illustration.'
            else:
                caption = IDENTITY + '. ' + rec['caption'].removeprefix('LQK4N. ') + ' Storybook illustration.'
            dst.with_suffix('.txt').write_text(caption + '\n')
            with Image.open(dst) as im:
                width, height = im.size
            records.append(dict(image=name, caption=caption, source=str(src), source_index=index,
                                crop_box=box, width=width, height=height,
                                sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
    manifest = dict(version='little-queen-flux2-klein-identity-v3', trigger='LQK4N',
                    identity_prefix=IDENTITY, image_count=len(records), unique_sources=len(KEEP),
                    strategy='20 selected originals plus 8 portrait crops; fresh base-model adapter, explicit identity anchor',
                    excluded_v2_indices=sorted(set(range(1,31))-set(KEEP)), images=records)
    (target/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (target/'metadata.jsonl').write_text(''.join(json.dumps(dict(file_name=r['image'],text=r['caption']))+'\n' for r in records))
    (target/'README.md').write_text('---\nlicense: other\n---\n# Little Queen v3\nPrivate synthetic identity dataset. 20 source images and 8 derived portrait crops, not 28 independent scenes.\n')
    review = HERE/'review'
    review.mkdir(exist_ok=True)
    portraits = [r for r in records if r['crop_box']]
    sheet = Image.new('RGB',(1024,552),'white')
    draw = ImageDraw.Draw(sheet)
    for i,r in enumerate(portraits):
        with Image.open(target/r['image']) as im:
            im.thumbnail((256,256))
            x,y=(i%4)*256,(i//4)*276
            sheet.paste(im,(x,y));draw.text((x,y+257),str(r['source_index']),fill='black')
    sheet.save(review/'portraits.jpg')
    print(f'Prepared {len(records)} examples from {len(KEEP)} originals: {target}')


if __name__ == '__main__':
    main()
