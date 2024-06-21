# TODO provisorial, find test split information in SpeaQ

import os 
import shutil
import h5py

# Only 100 images from SpeaQ's SGG test set are in VG_100K, indicating other potential data sources
# Using the train-test split of
# "Visual Translation Embedding Network for Visual Relation Detection"
# Zhang et al. 2017 https://arxiv.org/abs/1702.08319 
# Source: https://drive.google.com/file/d/1C6MDiqWQupMrPOgk4T12zWiAJAZmY1aa/view (file owner is Zawlin Kyaw)
# Recommended in: https://github.com/yangxuntu/vrd
sgg_path = '/mnt/orca/visual_genome/dataset/vg1_2_meta.h5'
vg_path = "/mnt/orca/visual_genome/dataset/images/VG_100K/"
target_dir = 'VG_100K_subset'

os.makedirs(target_dir, exist_ok=True)

with h5py.File(sgg_path, 'r') as f:
    test_ids = list(f["gt"]["test"].keys())

existing_files = [
    os.path.join(vg_path, f"{id}.jpg") for id in test_ids
    if os.path.isfile(os.path.join(vg_path, f"{id}.jpg"))
]

for file_path in existing_files:
    shutil.copy2(file_path, target_dir)

# Number of copied images
size = len(existing_files)

print(f"Number of images in test split: {size}")