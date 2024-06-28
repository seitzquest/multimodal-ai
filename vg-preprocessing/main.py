import os
import random
import argparse
from datasets import load_dataset
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

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


# Uses the cooccurence matrix to find the least likely object in the image and most likely object not in the image
def find_correlated_object(cooccurrence_matrix, objects_in_image):
    likelihoods = []
    for obj in objects_in_image:
        likelihood = cooccurrence_matrix.loc[obj, objects_in_image].sum()
        likelihoods.append(likelihood)
    
    # Find the least likely object in the image
    least_likely_idx = np.argmin(likelihoods)
    least_likely_object = objects_in_image[least_likely_idx]

    # Find the best replacement object
    remaining_objects = objects_in_image[:least_likely_idx] + objects_in_image[least_likely_idx+1:]
    best_replacement_object = None
    best_replacement_score = -1

    for obj in cooccurrence_matrix.index:
        if obj not in objects_in_image:
            replacement_score = cooccurrence_matrix.loc[obj, remaining_objects].sum()
            if replacement_score > best_replacement_score:
                best_replacement_score = replacement_score
                best_replacement_object = obj

    return least_likely_object, best_replacement_object

def extract_subimage(img, patch):
    x, y, w, h = patch
    return img.crop((x, y, x + w, y + h))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, default="VG_100K_subset", help="Directory containing images to modify")
    parser.add_argument("--modified_directory", type=str, default="VG_100K_subset_modified", help="Directory to save modified images")
    parser.add_argument("--overlay_image_path", type=str, default="insert_objects/maikaefer.png", help="Path to overlay image")
    parser.add_argument("--seed", type=int, help="Seed for random number generator")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to process")
    parser.add_argument("--patch_strategy", type=str, default="minimal", help="Strategy for finding patch. Can be 'minimal', 'maximal' or 'random'")
    parser.add_argument("--visualize-bb", action="store_true", help="Visualize bounding boxes")
    parser.add_argument("--correlate_overlay", action="store_true", help="Overrides the overlay image and patch strategy by using a correlated object")
    return parser.parse_args()

def get_patch(object):
    return object['x'], object['y'], object['w'], object['h']


def main():
    print("----------------------------")
    print("Visual Genome Preprocessing")
    print("----------------------------\n")
    args = parse_args()

    modified_directory = args.modified_directory
    overlay_image_path = args.overlay_image_path
    seed = args.seed
    patch_strategy = args.patch_strategy
    number_of_images = args.num_images
    visualize_bb = args.visualize_bb
    correlate_overlay = args.correlate_overlay

    os.makedirs(modified_directory, exist_ok=True)

    if seed is not None:
        print(f"Using set seed {seed}")
        # Set the seed of the random number generator for consistency and reproducibility
        random.seed(seed)

    dataset = load_dataset("visual_genome", "objects_v1.2.0")

    # Shuffle the dataset and select a subset of images
    # TODO: Test split -> simply add our code into SpeaQ/etc eval code to get the testset
    dataset = dataset.shuffle(seed=seed)['train'].select(range(number_of_images))

    if correlate_overlay:
        print(f"Collecting object names...")

        name_to_best_res_imgpatch = {}
        for example in tqdm(dataset):
            for obj in example['objects']:
                for name in obj['names']:
                    if name not in name_to_best_res_imgpatch or obj['w'] * obj['h'] > name_to_best_res_imgpatch[name][1][0] * name_to_best_res_imgpatch[name][1][1]:
                        name_to_best_res_imgpatch[name] = (example["image"], get_patch(obj))
        
        names = list(name_to_best_res_imgpatch.keys())
        print(f"Found {len(names)} unique object names.")

        cooccurence_matrix = pd.DataFrame(0, index=names, columns=names)

        print(f"Analyzing co-occurences...")
        for example in tqdm(dataset):
            objects = [name for obj in example['objects'] for name in obj['names']]
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    cooccurence_matrix.at[objects[i], objects[j]] += 1
                    cooccurence_matrix.at[objects[j], objects[i]] += 1
    else:
        print(f"Using overlay image {overlay_image_path}")
        overlay_image = Image.open(overlay_image_path)

    print(f"Processing {number_of_images} images...")
    for example in tqdm(dataset):
        img = example['image'].copy()

        if correlate_overlay:
            objects = {name: obj for obj in example['objects'] for name in obj['names']}
            least_likely_object, best_replacement_object = find_correlated_object(cooccurence_matrix, list(objects.keys()))
            overlay_image = extract_subimage(*name_to_best_res_imgpatch[best_replacement_object])
            
            replaced_object = objects[least_likely_object]
            x, y, w, h = replaced_object['x'], replaced_object['y'], replaced_object['w'], replaced_object['h']
            scaled_overlay = overlay_image.resize((w, h)).convert("RGBA")

            patch = (x, y, w, h)
        else:
            rotation = random.randint(0, 360)
            rotated_overlay = rotate_image_pillow(overlay_image, rotation)
            scaled_overlay = scale_inpainted_image_pillow(img, rotated_overlay)

            # Find the bounding boxes of the objects in the image
            overlaps = find_bounding_boxes_area(img.height, img.width, example['objects'])

            # Find the patch where the overlay image should be placed
            # This can be done using 3 different methods
            # 1. Find the patch with the minimal overlap of bounding boxes, tries to occlude as few objects as possible
            # 2. Find the patch with the maximal overlap of bounding boxes, tries to occlude as many objects as possible
            # 3. Find a random patch where the average overlap of bounding boxes is within one standard deviation of the mean
            if patch_strategy == "minimal":
                patch = find_minimal_patch(overlaps, scaled_overlay.size)
            elif patch_strategy == "maximal":
                patch = find_maximal_patch(overlaps, scaled_overlay.size)
            elif patch_strategy == "random":
                patch = find_random_thresholded_patch(overlaps, scaled_overlay.size)
            else:
                raise ValueError(f"Unknown patch strategy {patch_strategy}")

            if visualize_bb:
                plt.imshow(overlaps)
                plt.show()

        img = add_transparent_image_pillow(img, scaled_overlay, x_offset=patch[0], y_offset=patch[1])
        img.save(f"{modified_directory}/{example['image_id']}.jpg")


    print(f"Saved modified images to {modified_directory}")
    print("Done! :)")

if __name__ == "__main__":
    main()

# TODO: 
# Type hints