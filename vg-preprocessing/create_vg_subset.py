# TODO provisorial, find test split information in SpeaQ

import os 
import shutil
import random

vg_path = "/mnt/orca/visual_genome/dataset/images/VG_100K/"
target_dir = 'VG_100K_subset'

# Create the target directory if it does not exist
os.makedirs(target_dir, exist_ok=True)

# List all files in the source directory
all_images = os.listdir(vg_path)

# Randomly select 100 images from the source directory
selected_images = random.sample(all_images, 10)

# Copy the selected images to the target directory
for image in selected_images:
    shutil.copy(os.path.join(vg_path, image), os.path.join(target_dir, image))