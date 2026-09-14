#!/bin/bash
set -e

# Run Prompt Generation
echo "Starting Prompt Generation..."
python3 /mnt/storage/projects/agentic/images/scripts/generate_prompts.py
echo "Prompt Generation complete."

# Sleep to ensure RAM is reclaimed
echo "Sleeping for 15 seconds to allow RAM to clear..."
sleep 15

# Run Image Generation
echo "Starting Image Generation..."
python3 /mnt/storage/projects/agentic/images/scripts/generate_images.py
echo "Image Generation complete."

# Sleep to ensure RAM is reclaimed
echo "Sleeping for 15 seconds to allow RAM to clear..."
sleep 15

# Run Upscaling
echo "Starting Upscaling..."
python3 /mnt/storage/projects/agentic/images/scripts/upscale_images.py
echo "Upscaling complete. Pipeline finished."
