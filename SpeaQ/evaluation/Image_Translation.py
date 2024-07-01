import cv2
import numpy as np
import os
import random
import argparse

from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import torch
import copy
from tqdm import tqdm
import pandas as pd
import json
import requests
#setup for segment anything
sam_checkpoint = "./SAM_checkpoint/sam_vit_h_4b8939.pth"
from segment_anything import sam_model_registry, SamPredictor
model_type = "vit_h"
#change the device to cuda if you have a gpu
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def draw_semantic_shape_without_Background(shape = "triangle"):
    # Create a black background
    height, width = 700, 700
    background = np.zeros((height, width, 4), dtype=np.uint8)

    # Define the vertices of the shape
    if shape == "triangle":
        vertices = np.array([[250, 100], [50, 600], [450, 600]], np.int32)
        vertices = vertices.reshape((-1, 1, 2))
        # cv2.polylines(background, [vertices], isClosed=True, color=(0, 0, 255), thickness=2)
        # Fill the triangle with red color
        cv2.fillPoly(background, [vertices], color=(0, 0, 255, 255))
    else:
        top_left_vertex = (150, 150)
        bottom_right_vertex = (350, 350)

        vertices = np.array([[100, 100], [100, 400], [400, 400], [400,100]], np.int32)
        vertices = vertices.reshape((-1, 1, 2))
        # Draw and fill the square with blue color (BGR format, blue is (255, 0, 0))
        background = cv2.rectangle(background, top_left_vertex, bottom_right_vertex, (0, 0, 255), -1)



    return background

def draw_semantic_shape_with_Background(shape = "triangle"):
    # Create a black background
    height, width = 500, 500
    background = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the vertices of the shape
    if shape == "triangle":
        vertices = np.array([[250, 100], [150, 400], [350, 400]], np.int32)
        vertices = vertices.reshape((-1, 1, 2))
        # cv2.polylines(background, [vertices], isClosed=True, color=(0, 0, 255), thickness=2)
        # Fill the triangle with red color
        cv2.fillPoly(background, [vertices], color=(0, 0, 255))
    else:
        top_left_vertex = (150, 150)
        bottom_right_vertex = (350, 350)
        vertices = np.array([[100, 100], [100, 300], [300, 300], [300,100]], np.int32)
        cv2.fillPoly(background, [vertices], color=(0, 0, 255))


    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    plt.figure(figsize=(20, 20))
    plt.imshow(background)
    plt.axis('off')
    plt.show()

    return background

def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    rotated_image = cv2.warpAffine(img, M, tuple(size_new.astype(int)))
    return rotated_image

def scale_inpainted_image(source_image, target_image, scaling=1):
    bg_h, bg_w = source_image.shape[:2]
    fg_h, fg_w = target_image.shape[:2]

    width_scaling = bg_w * scaling / fg_w
    height_scaling = bg_h * scaling / fg_h
    scale_factor = min(width_scaling, height_scaling)

    scaled_width = int(fg_w * scale_factor)
    scaled_height = int(fg_h * scale_factor)

    scaled_foreground = cv2.resize(target_image, (scaled_width, scaled_height))
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

    return background

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


def find_bounding_boxes_area(height, width, objects):
    overlap_array = np.zeros((height, width), dtype=int)
    for o in objects:
        x1, y1, x2, y2 = o
        overlap_array[y1:y2, x1:x2] += 1

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

# Finds the image patch with the minimal overlap of bounding boxes
# Uses a heuristic to find the minimal patch
# Assumes that the minimal patch has the minimal value in the overlap array
def find_minimal_patch_heuristic(overlap_array, patch_size):
    min_sum = float('inf')
    min_patch = None

    # find positions where value is minimal and still can fit the patch
    cropped_array = overlap_array[:overlap_array.shape[0] - patch_size[0] + 1, :overlap_array.shape[1] - patch_size[1] + 1].copy()
    min_positions = np.argwhere(cropped_array == np.min(cropped_array))

    if min_positions.size == 0:
        return None

    # iterate through each of the minimal positions and find the one with the least sum
    for y, x in min_positions:
        patch = overlap_array[y:y+patch_size[0], x:x+patch_size[1]]
        patch_sum = np.sum(patch)
        if patch_sum < min_sum:
            min_sum = patch_sum
            min_patch = (x, y, patch_size[1], patch_size[0])

    return min_patch

def find_maximal_patch_heuristic(overlap_array, patch_size):
    max_sum = 0
    max_patch = None

    # find positions where value is maximal and still can fit the patch
    cropped_array = overlap_array[:overlap_array.shape[0] - patch_size[1], :overlap_array.shape[1] - patch_size[0]].copy()
    max_positions = np.argwhere(cropped_array == np.max(cropped_array))

    if max_positions.size == 0:
        return None

    # iterate through each of the maximal positions and find the one with the least sum
    for y, x in max_positions:
        patch = overlap_array[y:y+patch_size[1], x:x+patch_size[0]]
        patch_sum = np.sum(patch)
        if patch_sum > max_sum:
            max_sum = patch_sum
            max_patch = (x, y, patch_size[0], patch_size[1])

    return max_patch

def find_random_thresholded_patch_heuristic(overlap_array, patch_size):
    upper_threshold = np.mean(overlap_array) + np.std(overlap_array)
    lower_threshold = np.mean(overlap_array) - np.std(overlap_array)

    valid_positions = (overlap_array > lower_threshold) & (overlap_array < upper_threshold)

    valid_positions = np.argwhere(valid_positions)
    if valid_positions.size == 0:
        return None

    x, y = random.choice(valid_positions)
    return (x, y, patch_size[1], patch_size[0])

def duplicate_object(inputs,  obj_in_rl = True, mode = "same object duplicate"):

    #change the image to the right format for processing
    img = inputs['image']
    segment_background = copy.deepcopy(img.numpy())
    segment_background = np.transpose(segment_background, axes=[1, 2, 0])

    #get the object and predicate information from json file
    vocab_file = json.load(open('data/datasets/VG/VG-SGG-dicts-with-attri.json'))
    idx2label = vocab_file['idx_to_label']
    object_labels = [idx2label[str(i + 1)] for i in inputs['instances'].get('gt_classes').tolist()]
    labels = ['{}-{}'.format(idx, idx2label[str(i + 1)]) for idx, i in
              enumerate(inputs['instances'].get('gt_classes').tolist())]
    idx2pred = vocab_file['idx_to_predicate']
    # get object labels from ground truth
    boxes = inputs['instances'].get('gt_boxes').tensor.tolist()
    # get relation labels from ground truth
    gt_rels = inputs['relations'].tolist()
    gt_rels_labels = [(labels[i[0]], idx2pred[str(i[2] + 1)], labels[i[1]]) for i in gt_rels]
    objects_in_rl = [i[0] for i in gt_rels] + [i[1] for i in gt_rels]
    objects_idx = [i for i in range(len(object_labels))]
    objects_not_in_rl = [i for i in objects_idx if i not in objects_in_rl]

    "the mode could be:  same object duplicate, same class different object"
    if obj_in_rl:
        "segment the first object listet in the objects_in_rl list"
        object = segment_object(segment_background, boxes[objects_in_rl[0]])
    else:
        "segment the first object listet in the objects_not_in_rl list"
        object = segment_object(segment_background, boxes[objects_not_in_rl[0]])
    return object



def segment_object(segment_background, boxes):
    # get the object from the image
    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes
    object_box = segment_background[int(y1):int(y2), int(x1):int(x2)]

    predictor.set_image(segment_background)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes[None, :],
        multimask_output=False,
    )
    binary_mask = (masks[0] * 255).astype(np.uint8)
    extracted_object = cv2.bitwise_and(segment_background, segment_background, mask=binary_mask)
    b, g, r = cv2.split(extracted_object)
    alpha = binary_mask
    rgba_image = cv2.merge([b, g, r, alpha])

    return rgba_image



def find_correlated_object(cooccurrence_matrix, objects_in_image):
    likelihoods = []
    for obj in objects_in_image:
        likelihood = cooccurrence_matrix.loc[obj, objects_in_image].sum()
        likelihoods.append(likelihood)

    # Find the least likely object in the image
    least_likely_idx = np.argmin(likelihoods)
    least_likely_object = objects_in_image[least_likely_idx]

    # Find the best replacement object
    remaining_objects = objects_in_image[:least_likely_idx] + objects_in_image[least_likely_idx + 1:]
    best_replacement_object = None
    best_replacement_score = -1

    for obj in cooccurrence_matrix.index:
        if obj not in objects_in_image:
            replacement_score = cooccurrence_matrix.loc[obj, remaining_objects].sum()
            if replacement_score > best_replacement_score:
                best_replacement_score = replacement_score
                best_replacement_object = obj

    return least_likely_object, best_replacement_object
def get_co_occurence_matrix(data_loader):
    "the input of the data_loader is the splited testdata for the vealuation"
    # get the object lists from json file
    vocab_file = json.load(open('data/datasets/VG/VG-SGG-dicts-with-attri.json'))
    idx2label = vocab_file['idx_to_label']
    object_labels_150 = [idx2label[str(i + 1)] for i in range(150)]
   # generate_150_objects_overlays(object_labels_150)
    cooccurence_matrix = pd.DataFrame(0, index=object_labels_150, columns=object_labels_150)
    for inputs in data_loader:
        object_labels = [idx2label[str(i + 1)] for i in inputs[0]['instances'].get('gt_classes').tolist()]
        for i in range(len(object_labels)):
            for j in range(i + 1, len(object_labels)):
                cooccurence_matrix.at[object_labels[i], object_labels[j]] += 1
                cooccurence_matrix.at[object_labels[j], object_labels[i]] += 1

    return cooccurence_matrix
def generate_150_objects_overlays(object_labels):
    # Define the paths to the Visual Genome dataset annotation files
    image_data = json.load(open('data/datasets/image_data.json'))
    objects_data = json.load(open('data/datasets/objects.json'))
    # Find an image containing an object labeled as "person"
    for object in object_labels:
        name = object
        person_image_id = None
        person_bounding_box = None
        for obj in objects_data:
            for obj_item in obj['objects']:
                if 'names' in obj_item and object in obj_item['names']:
                    person_image_id = obj['image_id']
                    person_bounding_box = obj_item['x'], obj_item['y'], obj_item['w'], obj_item['h']
                    break
            if person_image_id:
                break

        if person_image_id is None:
            raise ValueError("No image with a " + object +" label found in the Visual Genome dataset.")

        if person_bounding_box is None:
            raise ValueError(f"No bounding box found for " + object +" in image_id {person_image_id}.")

        # Find the corresponding image metadata
        person_image_metadata = next((img for img in image_data if img['image_id'] == person_image_id), None)

        if person_image_metadata is None:
            raise ValueError(f"No metadata found for image_id {person_image_id}.")

        # Construct the URL for the image
        person_image_url = person_image_metadata['url']

        # Download the image
        response = requests.get(person_image_url)
        if response.status_code != 200:
            raise ValueError("Failed to download image from the Visual Genome dataset.")

        # Convert the image data to a NumPy array and then to an OpenCV image
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # segment the bounding box on the image
        x, y, w, h = person_bounding_box
        boxes = [x, y, x + w, y + h]
        obj_in_image = image[y:y + h, x:x + w]
        segmented_object = segment_object(image, boxes)


        # Save the image with the bounding box (optional)
        cv2.imwrite('evaluation/insert_objects/'+name+'_without_bounding_box.jpg', segmented_object)




def select_object(inputs, obj_in_rl = False, mode = None, co_occurence_matrix = None):
    # get the object lists from json file

    # change the image to the right format for processing
    img = inputs['image']
    segment_background = copy.deepcopy(img.numpy())
    segment_background = np.transpose(segment_background, axes=[1, 2, 0])

    # get the object and predicate information from json file
    vocab_file = json.load(open('data/datasets/VG/VG-SGG-dicts-with-attri.json'))
    idx2label = vocab_file['idx_to_label']
    object_labels = [idx2label[str(i + 1)] for i in inputs['instances'].get('gt_classes').tolist()]
    least_likely_object, best_replacement_object = find_correlated_object(co_occurence_matrix, object_labels)
   # translated_obj = cv2.imread('evaluation/insert_objects/'+best_replacement_object+'_without_bounding_box.jpg', cv2.IMREAD_UNCHANGED)
    translated_obj = cv2.imread('evaluation/insert_objects/airplane_without_bounding_box.jpg', cv2.IMREAD_UNCHANGED)
    translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
    return translated_obj




def image_translanting(inputs, mode = "trained_object", matrix = occurence_matrix):

    img = inputs['image']

    background_img = copy.deepcopy(img.numpy())
    background_img = np.transpose(background_img, axes=[1, 2, 0])

    if mode == "untrained_object":
        translated_obj = cv2.imread('evaluation/insert_objects/maikaefer.png', cv2.IMREAD_UNCHANGED)
        translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
    elif mode == "shape":
        translated_obj = draw_semantic_shape_without_Background(shape = "square")
    elif mode == "trained_object":
        translated_obj = cv2.imread('evaluation/insert_objects/aiplane.png', cv2.IMREAD_UNCHANGED)
        translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
    elif mode == "related_object_in_image":
        translated_obj = duplicate_object(inputs, obj_in_rl = False)
    elif mode == "likely_object_in_image":
        translated_obj = select_object(inputs, obj_in_rl = False, mode = "same class different object", matrix = matrix)

    elif mode == "unlikely_onject_in_image":
        translated_obj = select_object(inputs, obj_in_rl = False, mode = "unlikely object", matrix = matrix)

    rotation = 0  #random.randint(0, 360)
    rotated_overlay = rotate_image(translated_obj , rotation)
    scaled_overlay = scale_inpainted_image(background_img, rotated_overlay, scaling=1)

    # Find the bounding boxes of the objects in the image
    background_img_height, background_img_width = background_img.shape[:2]
    overlaps = find_bounding_boxes_area(background_img_height, background_img_width, inputs['instances'].get('gt_boxes').tensor.tolist())
    patch_strategy = "minimal"
    if patch_strategy == "minimal":
        patch = find_minimal_patch(overlaps, scaled_overlay.shape[:2])
    elif patch_strategy == "maximal":
        patch = find_maximal_patch(overlaps, scaled_overlay.size)
    elif patch_strategy == "random":
        patch = find_random_thresholded_patch(overlaps, scaled_overlay.size)
    elif patch_strategy == "minimal_heuristic":
        patch = find_minimal_patch_heuristic(overlaps, scaled_overlay.shape[:2])
    elif patch_strategy == "maximal_heuristic":
        patch = find_maximal_patch_heuristic(overlaps, scaled_overlay.shape[:2])
    elif patch_strategy == "random_heuristic":
        patch = find_random_thresholded_patch_heuristic(overlaps, scaled_overlay.shape[:2])
    else:
        raise ValueError(f"Unknown patch strategy {patch_strategy}")
    img_inpainting = add_transparent_image(background_img, scaled_overlay, patch[0], patch[1], rotation=rotation)

    plt.figure(figsize=(20, 20))
    plt.imshow(img_inpainting)
    plt.axis('off')
    plt.show()
    img_inpainting = np.transpose(img_inpainting, axes=[2, 0, 1])
    inputs['image'] =  torch.from_numpy(img_inpainting)


    return inputs

