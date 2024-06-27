import cv2
import numpy as np
import os
import random
import argparse
from datasets import load_dataset
from PIL import Image

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

def add_transparent_image_pillow(background, foreground, x_offset=None, y_offset=None, rotation=0):

    assert background.mode == 'RGB', f'background image should have exactly 3 channels (RGB). found:{background.mode}'
    assert foreground.mode == 'RGBA', f'foreground image should have exactly 4 channels (RGBA). found:{foreground.mode}'

    # center by default
    if x_offset is None:
        x_offset = (background.width - foreground.width) // 2
    if y_offset is None:
        y_offset = (background.height - foreground.height) // 2

    rotated_foreground = foreground.rotate(rotation, expand=True)

    position = (x_offset, y_offset)
    background.paste(rotated_foreground, position, rotated_foreground)
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


def find_valid_overlay_offsets(img_shape, overlay_shape, bounding_boxes):
    """
    Finds valid offsets for placing an overlay image on a background image,
    ensuring that the overlay does not overlap with any of the given bounding boxes.

    Args:
    - img_shape: Shape of the background image (height, width).
    - overlay_shape: Shape of the overlay image (height, width).
    - bounding_boxes: List of bounding boxes in the format (x, y, w, h).
                       Multiple bounding boxes can be provided.

    Returns:
    - x_offset: Valid x offset for placing the overlay.
    - y_offset: Valid y offset for placing the overlay.
    - success: Boolean indicating if a valid placement was found.
    """

    img_height, img_width = img_shape
    overlay_height, overlay_width = overlay_shape

    # Generate all potential positions where overlay can be placed
    potential_positions = []

    for y in range(img_height - overlay_height + 1):
        for x in range(img_width - overlay_width + 1):
            overlay_box = (x, y, overlay_width, overlay_height)
            overlap = False
            for bb in bounding_boxes:
                if boxes_overlap(bb, overlay_box):
                    overlap = True
                    break
            if not overlap:
                potential_positions.append((x, y))

    if potential_positions:
        # Randomly choose one of the valid positions
        x_offset, y_offset = random.choice(potential_positions)
        return x_offset, y_offset, True
    else:
        print("Could not find valid overlay placement.")
        return 0, 0, False

print("----------------------------")
print("Visual Genome Preprocessing")
print("----------------------------\n")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source_directory", type=str, default="VG_100K_subset", help="Directory containing images to modify")
parser.add_argument("--modified_directory", type=str, default="VG_100K_subset_modified", help="Directory to save modified images")
parser.add_argument("--overlay_image_path", type=str, default="insert_objects/maikaefer.png", help="Path to overlay image")
parser.add_argument("--seed", type=int, help="Seed for random number generator")

args = parser.parse_args()

source_directory = args.source_directory
modified_directory = args.modified_directory
overlay_image_path = args.overlay_image_path
vg_bounding_boxes_path = '/mnt/orca/visual_genome/dataset/VG-SGG-dicts-with-attri.json'
seed = args.seed
number_of_images = 10

os.makedirs(modified_directory, exist_ok=True)

print(f"Reading images from {source_directory}")
print(f"Using overlay image {overlay_image_path}")

if seed is not None:
    print(f"Using set seed {seed}")

# Seed the random number generator for consistency and reproducibility
random.seed(seed)

overlay = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
overlay_image = Image.open(overlay_image_path)

dataset = load_dataset("visual_genome", "objects_v1.2.0")
dataset = dataset.shuffle(seed=seed)['train'].select(range(number_of_images))

print("Processing images...")

for example in dataset:
    img = example['image'].copy()
    scaled_overlay = scale_inpainted_image_pillow(img, overlay_image)
    img = add_transparent_image_pillow(img, scaled_overlay, rotation=45)
    img.save(f"{modified_directory}/{example['image_id']}.jpg")

# for image_name in os.listdir(source_directory):
#     background = cv2.imread(os.path.join(source_directory, image_name))
#     img = background.copy()
    
#     rotation = random.randint(0, 360)
#     rotated_overlay = rotate_image(overlay, rotation)

#     scaled_overlay = scale_inpainted_image(img, rotated_overlay)

#     x_offset = random.randint(0, img.shape[1] - scaled_overlay.shape[1])
#     y_offset = random.randint(0, img.shape[0] - scaled_overlay.shape[0])

#     #x_offset, y_offset, possible = find_valid_overlay_offsets(img.shape[:2], scaled_overlay.shape[:2], [(0, 0, 10, 10)])
#     #if not possible:
#     #    continue

#     add_transparent_image(img, scaled_overlay, x_offset, y_offset, rotation)
#     modified_image_path = os.path.join(modified_directory, f"{os.path.splitext(image_name)[0]}_modified.jpg")
#     cv2.imwrite(modified_image_path, img)

print(f"Saved modified images to {modified_directory}")
print("Done! :)")

# TODO: 
# Test split in create_vg_subset
# Object-aware insertion