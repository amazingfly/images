import os
import shutil
import logging
from pathlib import Path
from PIL import Image
import torch
import timm
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
OUTPUT_DIR = Path("/mnt/storage/projects/agentic/images/scripts/output")
SFW_DIR = OUTPUT_DIR / "sfw"
NSFW_DIR = OUTPUT_DIR / "nsfw"

# Models
GENERAL_NSFW_MODEL = "Falconsai/nsfw_image_detection"
ANIME_TAGGER_MODEL = "hf_hub:SmilingWolf/wd-swinv2-tagger-v3"
MARQO_NSFW_MODEL = "Marqo/nsfw-image-detection-384"

class NSFWClassifier:
    def __init__(self):
        logger.info(f"Loading general NSFW model: {GENERAL_NSFW_MODEL}")
        self.general_classifier = pipeline("image-classification", model=GENERAL_NSFW_MODEL)
        
        logger.info(f"Loading Marqo NSFW model: {MARQO_NSFW_MODEL}")
        self.marqo_classifier = pipeline("image-classification", model=MARQO_NSFW_MODEL)

        logger.info(f"Loading anime tagger model: {ANIME_TAGGER_MODEL}")
        self.anime_model = timm.create_model(ANIME_TAGGER_MODEL, pretrained=True)
        self.anime_model.eval()
        
        # Get transforms for anime model
        self.data_config = timm.data.resolve_data_config(self.anime_model.pretrained_cfg)
        self.anime_transform = timm.data.create_transform(**self.data_config)
        
        self.ratings = ["general", "sensitive", "questionable", "explicit"]

    def is_nsfw(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            
            # 1. Falconsai Check
            general_results = self.general_classifier(img)
            nsfw_score = next((res['score'] for res in general_results if res['label'] == 'nsfw'), 0)
            
            # 2. Marqo Check
            marqo_results = self.marqo_classifier(img)
            # Marqo labels are usually 'nsfw' and 'sfw' or 'normal'
            marqo_nsfw_score = next((res['score'] for res in marqo_results if res['label'].lower() == 'nsfw'), 0)

            # 3. Anime Tagger Check
            input_tensor = self.anime_transform(img).unsqueeze(0)
            with torch.no_grad():
                output = self.anime_model(input_tensor)
                probs = torch.sigmoid(output)[0]
                rating_probs = probs[:4]
                rating_idx = torch.argmax(rating_probs).item()
                rating_label = self.ratings[rating_idx]
                
                s_score = rating_probs[1].item()
                q_score = rating_probs[2].item()
                e_score = rating_probs[3].item()

            # DECISION LOGIC
            # If any model is very sure, it's NSFW
            if nsfw_score > 0.5:
                return True, f"Falconsai NSFW: {nsfw_score:.3f}"
            if marqo_nsfw_score > 0.5:
                return True, f"Marqo NSFW: {marqo_nsfw_score:.3f}"
            if e_score > 0.2:
                return True, f"Anime Explicit: {e_score:.3f}"
            if q_score > 0.4:
                return True, f"Anime Questionable: {q_score:.3f}"
            
            # Cautionary combination
            # If it's sensitive AND either other model sees something suspicious
            if s_score > 0.5 and (nsfw_score > 0.1 or marqo_nsfw_score > 0.1):
                return True, f"Suggestive Anime + Sus: S={s_score:.2f}, F={nsfw_score:.2f}, M={marqo_nsfw_score:.2f}"

            # If it's questionable but not quite 0.4, but others see it
            if q_score > 0.2 and (nsfw_score > 0.1 or marqo_nsfw_score > 0.1):
                return True, f"Provocative Anime + Sus: Q={q_score:.2f}, F={nsfw_score:.2f}, M={marqo_nsfw_score:.2f}"

            # Final safety check: if general score is low across the board
            # and it's anime-sensitive, let's keep it in SFW if both general models say it's clean.
            
            return False, f"SFW (G:{rating_label}, F:{nsfw_score:.2f}, M:{marqo_nsfw_score:.2f}, S:{s_score:.2f})"
            
        except Exception as e:
            logger.error(f"Error classifying {img_path}: {e}")
            return True, f"Error: {e} (assuming NSFW for safety)"

def main(classifier, target_dir=None):
    if target_dir is None:
        target_dir = OUTPUT_DIR
    else:
        target_dir = Path(target_dir)

    sfw_dir = target_dir / "sfw"
    nsfw_dir = target_dir / "nsfw"

    # Move images back if they were moved (optional, but helpful for re-runs)
    if sfw_dir.exists():
        for f in sfw_dir.iterdir():
            if f.is_file(): shutil.move(f, target_dir / f.name)
    if nsfw_dir.exists():
        for f in nsfw_dir.iterdir():
            if f.is_file(): shutil.move(f, target_dir / f.name)

    sfw_dir.mkdir(exist_ok=True)
    nsfw_dir.mkdir(exist_ok=True)

    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    images = sorted([f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])
    
    logger.info(f"Found {len(images)} images to classify in {target_dir}.")

    for img_path in images:
        is_nsfw, reason = classifier.is_nsfw(img_path)
        dest = nsfw_dir if is_nsfw else sfw_dir
        shutil.move(img_path, dest / img_path.name)
        logger.info(f"Moved {img_path.name} to {dest.name}. Reason: {reason}")

if __name__ == "__main__":
    import sys
    base_output = Path("/mnt/storage/projects/agentic/images/scripts/output")
    
    classifier_instance = NSFWClassifier()
    
    # Process base output
    main(classifier_instance, base_output)
    # Process upscaled if it exists
    upscaled = base_output / "upscaled"
    if upscaled.exists():
        main(classifier_instance, upscaled)
