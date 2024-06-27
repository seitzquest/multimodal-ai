import cv2
import numpy as np
import os
import random
import argparse
from datasets import load_dataset
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Source: https://stackoverflow.com/a/71701023 
def add_transparent_image(background, foreground, x_offset=None, y_offset=None, rotation=0):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def add_transparent_image_pillow(background, foreground, x_offset=None, y_offset=None):

    assert background.mode == 'RGB', f'background image should have exactly 3 channels (RGB). found:{background.mode}'
    assert foreground.mode == 'RGBA', f'foreground image should have exactly 4 channels (RGBA). found:{foreground.mode}'

    # center by default
    if x_offset is None:
        x_offset = (background.width - foreground.width) // 2
    if y_offset is None:
        y_offset = (background.height - foreground.height) // 2

    position = (x_offset, y_offset)
    background.paste(foreground, position, foreground)
    return background

def scale_inpainted_image(source_image, target_image, scaling=0.2):
    bg_h, bg_w = source_image.shape[:2]
    fg_h, fg_w = target_image.shape[:2]

    width_scaling = bg_w * scaling / fg_w
    height_scaling = bg_h * scaling / fg_h
    scale_factor = min(width_scaling, height_scaling)

    scaled_width = int(fg_w * scale_factor)
    scaled_height = int(fg_h * scale_factor)

    scaled_foreground = cv2.resize(target_image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    return scaled_foreground

def scale_inpainted_image_pillow(source_image, target_image, scaling=0.2):
    bg_w, bg_h = source_image.size
    fg_w, fg_h = target_image.size

    width_scaling = bg_w * scaling / fg_w
    height_scaling = bg_h * scaling / fg_h
    scale_factor = min(width_scaling, height_scaling)

    scaled_width = int(fg_w * scale_factor)
    scaled_height = int(fg_h * scale_factor)

    scaled_foreground = target_image.resize((scaled_width, scaled_height))
    return scaled_foreground

def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

def rotate_image_pillow(img, angle):
    return img.rotate(angle, expand=True)


# for visualizing bounding boxes
def draw_bounding_box_pillow(image, bounding_box, color=(0, 255, 0), width=2):
    draw = ImageDraw.Draw(image)
    x, y, w, h = bounding_box
    draw.rectangle([x, y, x + w, y + h], fill=color)
    return image

# Returns a heatmap which shows where bounding boxes are
def find_bounding_boxes_area(height, width, objects):
    overlap_array = np.zeros((height, width), dtype=int)
    for o in objects:
        x, y, w, h = o['x'], o['y'], o['w'], o['h']
        overlap_array[y:y+h, x:x+w] += 1
    
    return overlap_array

# Finds the image patch with the minimal overlap of bounding boxes
def find_minimal_patch(overlap_array, patch_size):
    min_sum = float('inf')
    min_patch = None

    # Sliding window approch for finding the minimal patch
    for y in range(overlap_array.shape[0] - patch_size[0] + 1):
        for x in range(overlap_array.shape[1] - patch_size[1] + 1):
            patch = overlap_array[y:y+patch_size[0], x:x+patch_size[1]]
            patch_sum = np.sum(patch)
            if patch_sum < min_sum:
                min_sum = patch_sum
                min_patch = (x, y, patch_size[1], patch_size[0])
    return min_patch


# Finds the image patch with the maximal overlap of bounding boxes
def find_maximal_patch(overlap_array, patch_size):
    max_sum = 0
    max_patch = None

    # Sliding window approch for finding the maximal patch
    for y in range(overlap_array.shape[0] - patch_size[0] + 1):
        for x in range(overlap_array.shape[1] - patch_size[1] + 1):
            patch = overlap_array[y:y+patch_size[0], x:x+patch_size[1]]
            patch_sum = np.sum(patch)
            if patch_sum > max_sum:
                max_sum = patch_sum
                max_patch = (x, y, patch_size[1], patch_size[0])
    return max_patch


# Finds a random patch where the average overlap of bounding boxes is within one standard deviation of the mean
def find_random_thresholded_patch(overlap_array, patch_size):
    averages_array = np.zeros((overlap_array.shape[0] - patch_size[0] + 1, overlap_array.shape[1] - patch_size[1] + 1))
    for y in range(overlap_array.shape[0] - patch_size[0] + 1):
        for x in range(overlap_array.shape[1] - patch_size[1] + 1):
            patch = overlap_array[y:y+patch_size[0], x:x+patch_size[1]]
            averages_array[y, x] = np.mean(patch)

    upper_threshold = np.mean(averages_array) + np.std(averages_array)
    lower_threshold = np.mean(averages_array) - np.std(averages_array)

    valid_positions = (averages_array > lower_threshold) & (averages_array < upper_threshold)

    valid_positions = np.argwhere(valid_positions)
    if valid_positions.size == 0:
        return None

    x, y = random.choice(valid_positions)
    return (x, y, patch_size[1], patch_size[0])


print("----------------------------")
print("Visual Genome Preprocessing")
print("----------------------------\n")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source_directory", type=str, default="VG_100K_subset", help="Directory containing images to modify")
parser.add_argument("--modified_directory", type=str, default="VG_100K_subset_modified", help="Directory to save modified images")
parser.add_argument("--overlay_image_path", type=str, default="insert_objects/maikaefer.png", help="Path to overlay image")
parser.add_argument("--seed", type=int, help="Seed for random number generator")
parser.add_argument("--num_images", type=int, default=100, help="Number of images to process")

args = parser.parse_args()

modified_directory = args.modified_directory
overlay_image_path = args.overlay_image_path
seed = args.seed

number_of_images = args.num_images

os.makedirs(modified_directory, exist_ok=True)

print(f"Using overlay image {overlay_image_path}")

if seed is not None:
    print(f"Using set seed {seed}")

# Seed the random number generator for consistency and reproducibility
random.seed(seed)

overlay_image = Image.open(overlay_image_path)

dataset = load_dataset("visual_genome", "objects_v1.2.0")

# Shuffle the dataset and select a subset of images
dataset = dataset.shuffle(seed=seed)['train'].select(range(number_of_images))

# Collect all unique object names
print(f"Collecting object names...")
names = set()

for example in tqdm(dataset):
    objects = [name for obj in example['objects'] for name in obj['names']]
    names.update(objects)

print(f"Found {len(names)} unique object names.")

names = list(names)

cooccurence_matrix = pd.DataFrame(0, index=names, columns=names)

print(f"Analyzing co-occurences...")

# Count co-occurences of objects
for example in tqdm(dataset):
    objects = [name for obj in example['objects'] for name in obj['names']]
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            cooccurence_matrix.at[objects[i], objects[j]] += 1
            cooccurence_matrix.at[objects[j], objects[i]] += 1

print(f"Processing {number_of_images} images...")

# Process each image
for example in tqdm(dataset):
    img = example['image'].copy()

    rotation = random.randint(0, 360)#
    rotated_overlay = rotate_image_pillow(overlay_image, rotation)
    scaled_overlay = scale_inpainted_image_pillow(img, rotated_overlay)

    # Find the bounding boxes of the objects in the image
    overlaps = find_bounding_boxes_area(img.height, img.width, example['objects'])

    # Find the patch where the overlay image should be placed
    # This can be done using 3 different methods
    # 1. Find the patch with the minimal overlap of bounding boxes, tries to occlude as few objects as possible
    # 2. Find the patch with the maximal overlap of bounding boxes, tries to occlude as many objects as possible
    # 3. Find a random patch where the average overlap of bounding boxes is within one standard deviation of the mean

    patch = find_minimal_patch(overlaps, scaled_overlay.size)
    #patch = find_maximal_patch(overlaps, scaled_overlay.size)
    #patch = find_random_thresholded_patch(overlaps, scaled_overlay.size)

    # for visualizing where many bounding boxes are
    #plt.imshow(overlaps)
    #plt.show()

    img = add_transparent_image_pillow(img, scaled_overlay, x_offset=patch[0], y_offset=patch[1])
    img.save(f"{modified_directory}/{example['image_id']}.jpg")


print(f"Saved modified images to {modified_directory}")
print("Done! :)")

# TODO: 
# Test split in create_vg_subset
# Create some sort of algorithm to find the appropriate new object from the co-occurence matrix based on the existing objects in the image