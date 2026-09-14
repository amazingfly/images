import json
import os
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upscale_images(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    os.makedirs(config['upscaled_dir'], exist_ok=True)
    
    # Process images in output_dir
    files = sorted([f for f in os.listdir(config['output_dir']) if f.endswith('.png')])
    
    for filename in files:
        input_path = os.path.join(config['output_dir'], filename)
        output_path = os.path.join(config['upscaled_dir'], filename)
        
        if os.path.exists(output_path):
            logger.info(f"Image {filename} already upscaled. Skipping.")
            continue
            
        logger.info(f"Upscaling {filename}...")
        
        # Read image
        img = cv2.imread(input_path)
        
        # Upscale to 1080p (roughly 1920x1080)
        # Lanczos is generally the best for quality upscaling without an external model
        upscaled_img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        
        # Save
        cv2.imwrite(output_path, upscaled_img)
        
        logger.info(f"Finished upscaling {filename}.")

if __name__ == "__main__":
    upscale_images("config.json")
