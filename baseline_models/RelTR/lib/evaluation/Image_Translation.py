# von Zoey's SpeaQ (geänderte Stellen werden noch mit #edit markiert) -> kann man auch einfach mit diff am Ende vergleichen

import cv2
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import json
import requests
from skimage.util import view_as_windows
from util.misc import NestedTensor
from util.box_ops import box_cxcywh_to_xyxy
from segment_anything import sam_model_registry, SamPredictor


def setup_sam_predictor(device):
    #activate segment ANYTHING
    sam_checkpoint = "./ckpt/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam_predictor = setup_sam_predictor(device)


def draw_semantic_shape_without_Background(shape = "triangle", height=700, width=700):
    """
        Generate the semantic shape without background

        Parameters:
            shape (string): the shape that should be generated.
                Can be either "triangle" or "square"


        Returns:
            numpy.ndarray: An array of shape (height, width, 4) of the generated picture.

    """
    if shape == "triangle":
        vertices = np.array([[250, 100], [50, 600], [450, 600]], np.int32)
        background = np.zeros((height, width, 4), dtype=np.uint8)
        cv2.fillPoly(background, [vertices], color=(0, 0, 255, 255))
    else:
        top_left_vertex = (150, 150)
        bottom_right_vertex = (350, 350)
        vertices = np.array([[100, 100], [100, 400], [400, 400], [400, 100]], np.int32)
        background = cv2.rectangle(np.zeros((height, width, 4), dtype=np.uint8), top_left_vertex, bottom_right_vertex, (0, 0, 255, 255), -1)
    return background


def draw_semantic_shape_with_Background(shape = "triangle", height=500, width=500):
    """
        Generate the semantic shape with black background

        Parameters:
            shape (string): the shape that should be generated.
                Can be either "triangle" or "square"


        Returns:
            numpy.ndarray: An array of shape (height, width, 4) of the generated picture.

    """
    # Create a black background
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
    #plt.figure(figsize=(20, 20))
    #plt.imshow(background)
    #plt.axis('off')
    #plt.show()
    return background


def rotate_image(img, angle):
    """
    Rotates the image by the given angle.

    Parameters:
        img (cv2.Image): The input image to be rotated.
        angle (float): The angle in degrees to rotate the image.

    Returns:
        cv2.Image: The rotated image.
    """
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    rotated_image = cv2.warpAffine(img, M, tuple(size_new.astype(int)))
    return rotated_image


def scale_inpainted_image(source_image, target_image, scaling=1):
    """
    Scales the target image to be a fraction of the source image.
    The scaling is based on the width and height of the source image, taking into account the smaller value of the two.

    Parameters:
        source_image (torch.tensor): The source image to scale the target image to.
        target_image (torch.tensor): The target image to be scaled.
        scaling (float, optional): The scaling factor. Defaults to 0.2.

    Returns:
        torch.tensor: The target image but scaled
    """
    bg_h, bg_w = source_image.shape[:2]
    fg_h, fg_w = target_image.shape[:2]

    width_scaling = bg_w * scaling / fg_w
    height_scaling = bg_h * scaling / fg_h
    scale_factor = min(width_scaling, height_scaling)

    scaled_width = int(fg_w * scale_factor)
    scaled_height = int(fg_h * scale_factor)

    target_tensor = target_image.float().permute(2, 0, 1).unsqueeze(0)
    scaled_foreground = F.interpolate(target_tensor, size=(scaled_height, scaled_width), mode='bilinear', align_corners=False)
    
    # Remove batch dimension and return as tensor
    scaled_foreground = scaled_foreground.squeeze(0).permute(1, 2, 0)
    return scaled_foreground


def add_transparent_image(background, foreground, x_offset=None, y_offset=None, rotation=0):
    """
    Adds a transparent foreground image onto a background image.

    Parameters:
        background (torch.tensor): The base image
        foreground (torch.tensor): The foreground image containing the to be transplanted object. Needs to have an alpha channel.
        x_offset (int, optional): The x-coordinate offset for the foreground image. If not provided, the foreground image will be centered horizontally. Defaults to None.
        y_offset (int, optional): The y-coordinate offset for the foreground image. If not provided, the foreground image will be centered vertically. Defaults to None.

    Returns:
        torch.tensor: The resulting image with the transplanted image.
    """
    
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return background

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3].float() / 255  # Convert to float and normalize to 0.0-1.0

    # construct an alpha_mask that matches the image shape, add the channels and overwrite the background
    alpha_mask = alpha_channel.unsqueeze(2).expand(-1, -1, 3)
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return background


#1 Returns a heatmap which shows where bounding boxes are
def find_bounding_boxes_area(height, width, objects):
    overlap_array = np.zeros((height, width), dtype=int)
    for o in objects:
        x1, y1, x2, y2 = o
        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)
        overlap_array[y:y+h, x:x+w] += 1
    return overlap_array


# Finds the image patch with the minimal overlap of bounding boxes
def find_minimal_patch(overlap_array, patch_size):
    """
    Finds the minimal patch in the given overlap array.
    This uses a sliding window approach, which finds the patch area where the sum of values is minimal.

    Parameters:
        overlap_array (numpy.ndarray): The heatmap array of bounding box overlaps.
        patch_size (tuple): The size of the patch.

    Returns:
        tuple: The coordinates and size of the minimal patch.
    """
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
    """
    Finds the patch where the sum of overlap values is maximal.
    Works similar to find_minimal_patch() but finds the maximal patch instead aka the patch
    where the sum of overlap values is maximal.

    Parameters:
        overlap_array (numpy.ndarray): The input array containing the overlap values.
        patch_size (tuple): The size of the patch to be considered.

    Returns:
        tuple: A tuple representing the coordinates and size of the maximal patch in the format (x, y, width, height).
    """
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


#1 Finds a random patch where the average overlap of bounding boxes is within one standard deviation of the mean
def find_random_thresholded_patch(overlap_array, patch_size):
    """
    Finds a random patch in the given overlap array.
    This uses a sliding window approach to calculate the average overlap of bounding boxes in each patch.

    Parameters:
        overlap_array (numpy.ndarray): The heatmap array of bounding box overlaps.
        patch_size (tuple): The size of the patch.

    Returns:
        tuple: The coordinates and size of the minimal patch.
    """
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


#1 Finds the image patch with the minimal overlap of bounding boxes
# Uses a heuristic to find the minimal patch
# Assumes that the minimal patch has the minimal value in the overlap array
def find_minimal_patch_heuristic(overlap_array, patch_size):
    """
    Finds the position of the minimal patch using a heuristic approach
    This is necessary because find_minimal_patch() is too slow :(
    The key assumption is that the minimal patch has the minimal value in the overlap array,
    which narrows down the search space.

    Parameters:
        overlap_array (numpy.ndarray): The heatmap array of bounding box overlaps.
        patch_size (tuple): The size of the patch.

    Returns:
        tuple: The coordinates and size of the minimal patch.
    """
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

#1
def find_maximal_patch_heuristic(overlap_array, patch_size):
    """
    Finds the maximal patch in the given overlap array using a heuristic approach.
    Similarly to find_minimal_patch_heuristic(), this function assumes that the maximal patch
    has the maximal value in the overlap array.

    Parameters:
        overlap_array (numpy.ndarray): The heatmap array of bounding box overlaps.
        patch_size (tuple): The size of the patch.

    Returns:
        tuple: The coordinates and size of the maximal patch.
    """
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

#1
def find_random_thresholded_patch_heuristic(overlap_array, patch_size):
    upper_threshold = np.mean(overlap_array) + np.std(overlap_array)
    lower_threshold = np.mean(overlap_array) - np.std(overlap_array)

    valid_positions = (overlap_array > lower_threshold) & (overlap_array < upper_threshold)

    valid_positions = np.argwhere(valid_positions)
    if valid_positions.size == 0:
        return None

    x, y = random.choice(valid_positions)
    return (x, y, patch_size[1], patch_size[0])


#1
def find_minimal_patch_window(overlap_array, patch_size):
    # Ensure the patch fits within the overlap array dimensions
    if overlap_array.shape[0] < patch_size[0] or overlap_array.shape[1] < patch_size[1]:
        return None

    # Find the minimum value in the valid region where the patch can fit
    cropped_height = overlap_array.shape[0] - patch_size[0] + 1
    cropped_width = overlap_array.shape[1] - patch_size[1] + 1
    cropped_array = overlap_array[:cropped_height, :cropped_width]

    # Use view_as_windows to create sliding windows of the patch size
    try:
        windows = view_as_windows(overlap_array, (patch_size[0], patch_size[1]))
    except NameError:
        raise ImportError("The view_as_windows function is not available. Please install scikit-image.")

    # Calculate the sum of each window (patch)
    patch_sums = windows.sum(axis=(2, 3))

    # Find the index of the minimum sum
    min_idx = np.unravel_index(np.argmin(patch_sums), patch_sums.shape)
    min_sum = patch_sums[min_idx]

    # Calculate the top-left corner of the minimal patch
    y, x = min_idx
    min_patch = (x, y, patch_size[1], patch_size[0])
    return min_patch
    
#1
def find_minimal_patch_convolve(overlap_array, patch_size):
    # Convert the input array to a PyTorch tensor
    overlap_tensor = torch.tensor(overlap_array, dtype=torch.float32)

    rows, cols = overlap_tensor.shape
    patch_height, patch_width = patch_size

    min_sum = float('inf')
    min_patch = None

    # Process patches in a row-by-row manner to manage memory usage
    for y in range(rows - patch_height + 1):
        for x in range(cols - patch_width + 1):
            # Extract the current patch and move it to the GPU
            patch = overlap_tensor[y:y + patch_height, x:x + patch_width].cuda()

            # Compute the sum of the current patch
            patch_sum = patch.sum().item()

            # Move the patch back to the CPU to free up GPU memory
            patch.cpu()

            # Update the minimum patch if the current patch sum is smaller
            if patch_sum < min_sum:
                min_sum = patch_sum
                min_patch = (x, y, patch_width, patch_height)

    return min_patch


def duplicate_object(inputs, targets, obj_in_rl = True, mode = "same object duplicate", matrix = None):
    """
        Generate the transplanted object which is related to one existing object in the image,
        either by duplicating the object or by choosing an object from the same class.

        Parameters:
            inputs (Dictionary): the input image and the object information
            obj_in_rl (bool): if the object should be related to an object in the relation list
            mode (string): the mode of the object generation.
                Can be either "same object duplicate" to duplicate one object
                or "same class different object" to choose an object from the same class


        Returns:
            torch.tensor: A tensor of shape (height, width, 4) of the generated picture.
    """
    
    #change the image to the right format for processing
    img = inputs.decompose()[0][0]
    segment_background = img.clone().detach()
    segment_background, a_min, a_max = rgbize_image(segment_background)

    #get the object and predicate information from json file
    vocab_file = json.load(open('./data/vg/VG-SGG-dicts-with-attri.json'))
    idx2label = vocab_file['idx_to_label']
    object_labels = [idx2label[str(i)] for i in targets[0]['labels'].tolist()]
    labels = ['{}-{}'.format(idx, idx2label[str(i)]) for idx, i in
              enumerate(targets[0]["labels"].tolist())]

    idx2pred = vocab_file['idx_to_predicate']
    # get object labels from ground truth
    boxes = targets[0]["boxes"].tolist()
    # get relation labels from ground truth
    gt_rels = targets[0]['rel_annotations'].tolist()
    gt_rels_labels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    objects_in_rl = [i[0] for i in gt_rels] + [i[1] for i in gt_rels]
    objects_idx = [i for i in range(len(object_labels))]
    objects_not_in_rl = [i for i in objects_idx if i not in objects_in_rl]

    "the mode could be:  same object duplicate, same class different object"
    if obj_in_rl:
        if mode == "same object duplicate":
            "segment the first object listet in the objects_in_rl list"
            translated_obj = segment_object(segment_background, boxes[objects_in_rl[0]])
            idx_black = np.where(
                (translated_obj[:, :, 0] == 0) & (translated_obj[:, :, 1] == 0) & (translated_obj[:, :, 2] == 0))
            translated_obj[idx_black[0], idx_black[1], 3] = 0
        elif mode == "same class different object":
            "choose the object from the segmented object which in the same class as the object in the relation list"
            obj_name = object_labels[objects_in_rl[0]]
            translated_obj = cv2.imread(
                './lib/evaluation/insert_objects/' + obj_name + '_without_bounding_box.jpg', cv2.IMREAD_UNCHANGED)
            translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
            idx_black = np.where((translated_obj[:, :, 0] == 0) & (translated_obj[:, :, 1] == 0) & (translated_obj[:, :, 2] == 0))
            translated_obj[idx_black[0], idx_black[1], 3] = 0
    else:
        if (len(objects_not_in_rl) == 0):
            obj_idx = objects_in_rl[-1]
        else:
            obj_idx = objects_not_in_rl[0]
        if mode == "same object duplicate":
            "segment the first object listet in the objects_in_rl list"
            translated_obj = segment_object(segment_background, boxes[obj_idx])
            idx_black = np.where(
                (translated_obj[:, :, 0] == 0) & (translated_obj[:, :, 1] == 0) & (translated_obj[:, :, 2] == 0))
            translated_obj[idx_black[0], idx_black[1], 3] = 0
        elif mode == "same class different object":
            "choose the object from the segmented object which in the same class as the object not in the relation list"

            obj_name = object_labels[obj_idx]
            translated_obj = cv2.imread(
                './lib/evaluation/insert_objects/' + obj_name + '_without_bounding_box.jpg', cv2.IMREAD_UNCHANGED)
            translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
            idx_black = np.where(
                (translated_obj[:, :, 0] == 0) & (translated_obj[:, :, 1] == 0) & (translated_obj[:, :, 2] == 0))
            translated_obj[idx_black[0],idx_black[1], 3] = 0
    return translated_obj


def segment_object(segment_background, boxes):
    """
        Generate the segmented object based on the box information using SAM

        Parameters:
            segment_background (torch.tensor): the background image that is used for segmenting the object
            boxes (list): the box inforamtion which is used for the segmentation
            sam_predictor (SamPredictor): the predictor for the segment anything model


        Returns:
            torch.tensor: An array of shape (height, width, 4) of the segmented obejct.
    """
    segment_background = segment_background.cpu().numpy().astype(np.uint8)

    # get the object from the image
    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes
    object_box = segment_background[int(y1):int(y2), int(x1):int(x2)]

    sam_predictor.set_image(segment_background)
    masks, _, _ = sam_predictor.predict(
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


def find_least_likely_object(objects, cooccurrence_matrix, objects_in_image):
    """
        Find the unlikely object based on the object list in image and cooccurence matrix

        Parameters:
            objects (list): the list of objects that should be considered
            cooccurrence_matrix (pandas.DataFrame): the cooccurence matrix of the objects
            objects_in_image (list): the list of objects in the image

        Returns:
            string: the name of the unlikely object
    """
    likelihoods = []
    for obj in objects:
        likelihood = cooccurrence_matrix.loc[obj, objects_in_image].sum()
        likelihoods.append(likelihood)

    least_likely_idx = np.argmin(likelihoods)
    return objects[least_likely_idx]


# Uses the cooccurence matrix to find the least likely object in the image and least likely object not in the image
def find_correlated_object(cooccurrence_matrix, objects_in_image):
    """
        Find the correlated object based on the object list in image and cooccurence matrix

        Parameters:
            cooccurrence_matrix (pandas.DataFrame): the cooccurence matrix of the objects
            objects_in_image (list): the list of objects in the image

        Returns:
            string: the name of the unlikely object
    """
    least_likely_image_object = find_least_likely_object(objects_in_image, cooccurrence_matrix, objects_in_image)

    remaining_objects = [obj for obj in objects_in_image if obj != least_likely_image_object]
    external_objects = [obj for obj in cooccurrence_matrix.index if obj not in objects_in_image]
    least_likely_external_object = find_least_likely_object(external_objects, cooccurrence_matrix, remaining_objects)

    return least_likely_image_object, least_likely_external_object


def get_co_occurence_matrix(data_loader):
    """
        Generate the co-occurence matrix for the VG 150 Dataset used in the inference

        Parameters:
            data_loader (DataLoader): the data loader for the test data

        Returns:
            pandas.DataFrame: The generated co-occurence matrix
    """
    # Load co-occurrence matrix from file
    cooccurence_matrix_file='./lib/evaluation/cooccurence_matrix.pkl'
    if os.path.exists(cooccurence_matrix_file):
        with open(cooccurence_matrix_file, 'rb') as f:
            cooccurence_matrix = pd.read_pickle(f)
    else:
        # Get object labels from data loader
        vocab_file = json.load(open('./data/vg/VG-SGG-dicts-with-attri.json'))
        idx2label = vocab_file['idx_to_label']
        object_labels_150 = [idx2label[str(i + 1)] for i in range(150)]
        # generate_150_objects_overlays(object_labels_150)
        cooccurence_matrix = pd.DataFrame(0, index=object_labels_150, columns=object_labels_150)
    
        for _, targets in tqdm(data_loader):
            object_labels = [idx2label[str(i)] for i in targets[0]['labels'].tolist()] # NOTE: modified to work with RelTR data_loader
            for i in range(len(object_labels)):
                for j in range(i + 1, len(object_labels)):
                    cooccurence_matrix.at[object_labels[i], object_labels[j]] += 1
                    cooccurence_matrix.at[object_labels[j], object_labels[i]] += 1
        # Save co-occurrence matrix to file
        cooccurence_matrix.to_pickle(cooccurence_matrix_file)
    return cooccurence_matrix
    

def generate_150_objects_overlays(object_labels):
    """
        Generate the segmented object for the 150 object classes in the VG 150 Dataset

        Parameters:
            object_labels (list): the list of object labels
    """

    # Define the paths to the Visual Genome dataset annotation files
    image_data = json.load(open('./data/vg/image_data.json'))
    objects_data = json.load(open('./data/vg/objects.json'))
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
        cv2.imwrite('./lib/evaluation/insert_objects/'+name+'_without_bounding_box.jpg', segmented_object)


def select_object(inputs, obj_in_rl=False, mode=None, co_occurence_matrix=None, targets=None):
    """
        Generate the transplanted object which is related to one existing object in the image,
        In this case, the most unlikely object is selected based on the co-occurrence matrix.

        Parameters:
            inputs (Dictionary): the input image and the object information
            matrix (pandas.Dataframe): the generated co-occurrence matrix for the VG 150 Dataset
            mode (string): the mode of the object generation. The mode in this case is "unlikely object"


        Returns:
            numpy.ndarray: An array of shape (height, width, 4) of the generated picture.
    """
    
    if mode == "unlikely object":
        # Get the object and predicate information from json file
        vocab_file = json.load(open('./data/vg/VG-SGG-dicts-with-attri.json'))
        idx2label = vocab_file['idx_to_label']
        if targets is not None:
            object_labels = [idx2label[str(i)] for i in targets[0]['labels'].tolist()]
        else:
            object_labels = []

        least_likely_object, least_likely_external_object = find_correlated_object(co_occurence_matrix, object_labels)
        image_path = f'./lib/evaluation/insert_objects/{least_likely_external_object}_without_bounding_box.jpg'
        #translated_obj = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        translated_obj = cv2.imread('./lib/evaluation/insert_objects/'+least_likely_external_object+'_without_bounding_box.jpg', cv2.IMREAD_UNCHANGED)
        translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
        idx_black = np.where(
            (translated_obj[:, :, 0] == 0) & (translated_obj[:, :, 1] == 0) & (translated_obj[:, :, 2] == 0))
        translated_obj[idx_black[0], idx_black[1], 3] = 0

    return translated_obj


def rgbize_image(array):
    array = array.permute(1, 2, 0)
    a_min = torch.min(array)
    a_max = torch.max(array)
    return (255*(array-a_min)/(a_max-a_min)), a_min, a_max

def undo_rgbize(array, a_min, a_max):
    array = array / 255
    array = array * (a_max - a_min) + a_min
    return array.permute(2, 0, 1)

def save_image(array, output_dir='./demo/test', name="test"):
    array = array.cpu().numpy().astype(np.uint8)
    image = Image.fromarray(array)
    image.save(f'{output_dir}/{name}.png')


def image_translanting(inputs, mode="trained_object", matrix=None, targets=None, scaling=0.2, patch_strategy="fully_random", obj_in_rl=False):
    """
        The obejct transplanting strategies for the image

        Parameters:
            inputs (Dictionary): the input image and the object information
            matrix (pandas.Dataframe): the generated co-occurrence matrix for the VG 150 Dataset
            mode (string): the mode of the object generation.
                'mode can be: related_object_in_image,similar_object_in_image,unlikely_onject_in_image,trained_object, untrained_object'
            patch_strategy (string): the patch strategy for the object transplanting
                'patch could be minimal_heuristic,maximal_heuristic,random_heuristic'
            obj_in_rl (bool): if the object should be related to an object in the relation list
            scaling (float): the scaling factor for the object


        Returns:
           NestedTensor: the modified input image with the transplanted object
    """
    
    img = inputs.decompose()[0][0]
    background_img = img.clone().detach()
    background_img, a_min, a_max = rgbize_image(background_img)
    # save_image(background_img)

    if mode == "untrained_object":
        translated_obj = cv2.imread('./lib/evaluation/insert_objects/maikaefer.png', cv2.IMREAD_UNCHANGED)
        if translated_obj is None:
            raise FileNotFoundError("Image for untrained_object not found.")
        translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
        translated_obj = torch.from_numpy(translated_obj).to(device)
        scaled_overlay = scale_inpainted_image(background_img, translated_obj, scaling=scaling)
    elif mode == "shape":
        translated_obj = draw_semantic_shape_without_Background(shape="square")
        translated_obj = torch.from_numpy(translated_obj).to(device)
        scaled_overlay = scale_inpainted_image(background_img, translated_obj, scaling=scaling)
    elif mode == "trained_object":
        translated_obj = cv2.imread('./lib/evaluation/insert_objects/aiplane.png', cv2.IMREAD_UNCHANGED)
        if translated_obj is None:
            raise FileNotFoundError("Image for trained_object not found.")
        translated_obj = cv2.cvtColor(translated_obj, cv2.COLOR_BGRA2RGBA)
        translated_obj = torch.from_numpy(translated_obj).to(device)
        scaled_overlay = scale_inpainted_image(background_img, translated_obj, scaling=scaling)

    elif mode == "related_object_in_image":
        matrix = pd.read_pickle('./lib/evaluation/cooccurence_matrix.pkl')
        scaled_overlay = duplicate_object(inputs, targets, obj_in_rl = obj_in_rl,mode = "same object duplicate", matrix = matrix)
        scaled_overlay = torch.from_numpy(scaled_overlay).to(device)

    elif mode == "similar_object_in_image":
        matrix = pd.read_pickle('./lib/evaluation/cooccurence_matrix.pkl')
        scaled_overlay = duplicate_object(inputs, targets, obj_in_rl = obj_in_rl,mode = "same class different object", matrix = matrix)
        scaled_overlay = torch.from_numpy(scaled_overlay).to(device)

    elif mode == "unlikely_object_in_image":
        matrix = pd.read_pickle('./lib/evaluation/cooccurence_matrix.pkl')
        scaled_overlay = select_object(inputs, obj_in_rl=obj_in_rl, mode="unlikely object", co_occurence_matrix=matrix, targets=targets)
        scaled_overlay = torch.from_numpy(scaled_overlay).to(device)
    else:
        raise ValueError(f"Unknown mode for image transplanting {mode}")

    # save_image(translated_obj, name="overlay")
    if patch_strategy == "random":
        patch = np.random.randint(0, 5, 2)
    else:
        # Find the bounding boxes of the objects in the image
        background_img_height, background_img_width = background_img.shape[:2]

        # Get bounding boxes from targets instead of inputs
        if targets is not None and len(targets) > 0 and 'boxes' in targets[0]:
            gt_boxes = targets[0]['boxes'].cpu().numpy().tolist()
        else:
            gt_boxes = []

        overlaps = find_bounding_boxes_area(background_img_height, background_img_width, gt_boxes)
        if patch_strategy == "minimal":
            patch = find_minimal_patch(overlaps, scaled_overlay.shape[:2])
        elif patch_strategy == "maximal":
            patch = find_maximal_patch(overlaps, scaled_overlay.shape[:2])
        elif patch_strategy == "random": # disabled for now
            patch = find_random_thresholded_patch(overlaps, scaled_overlay.shape[:2])
        elif patch_strategy == "minimal_heuristic":
            patch = find_minimal_patch_window(overlaps, scaled_overlay.shape[:2])
        elif patch_strategy == "maximal_heuristic":
            patch = find_maximal_patch_heuristic(overlaps, scaled_overlay.shape[:2])
        elif patch_strategy == "random_heuristic":
            patch = find_random_thresholded_patch_heuristic(overlaps, scaled_overlay.shape[:2])
        else:
            raise ValueError(f"Unknown patch strategy {patch_strategy}")

    img_inpainting = add_transparent_image(background_img, scaled_overlay, patch[0], patch[1], rotation=0)

    # save_image(background_img, name="background")
    # save_image(scaled_overlay, name="overlay")
    # save_image(img_inpainting, name="modified")
    
    img_inpainting = undo_rgbize(img_inpainting, a_min, a_max)
    img_inpainting = img_inpainting.unsqueeze(dim=0)
    mask = inputs.decompose()[1]
    return NestedTensor(img_inpainting, mask)
