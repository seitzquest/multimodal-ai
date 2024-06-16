import cv2
import numpy as np
import os
import random

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

def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))


source_directory = "VG_100K_subset"
modified_directory = "VG_100K_subset_modified"
overlay_image_path = 'insert_objects/maikaefer.png'

os.makedirs(modified_directory, exist_ok=True)

overlay = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
for image_name in os.listdir(source_directory):
    background = cv2.imread(os.path.join(source_directory, image_name))
    img = background.copy()
    scaled_overlay = scale_inpainted_image(img, overlay)

    rotation = random.randint(0, 360)
    rotated_overlay = rotate_image(scaled_overlay, rotation)

    x_offset = random.randint(0, img.shape[1] - rotated_overlay.shape[1])
    y_offset = random.randint(0, img.shape[0] - rotated_overlay.shape[0])
    
    add_transparent_image(img, rotated_overlay, x_offset, y_offset, rotation)
    modified_image_path = os.path.join(modified_directory, f"{os.path.splitext(image_name)[0]}_modified.jpg")
    cv2.imwrite(modified_image_path, img)



# TODO: Test split in create_vg_subset


# Potential optimizations
# TODO: Calculate Scaling not only based on width, but maybe on height as well?