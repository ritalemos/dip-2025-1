# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate_img(img: np.ndarray) -> np.ndarray:
    img_height, img_width = img.shape
    shift_x = int(0.2 * img_width)  
    shift_y = int(0.2 * img_height)  
    
    translated_img = np.zeros_like(img)
    translated_img[shift_y:, shift_x:] = img[:img_height-shift_y, :img_width-shift_x]
    
    return translated_img

def stretch_img(img: np.ndarray) -> np.ndarray:
    img_height, img_width = img.shape
    new_width = int(img_width * 1.5)
    stretched_img = np.zeros((img_height, new_width), dtype=img.dtype)

    for row_idx in range(img_height):
        for col_idx in range(new_width):
            source_col = int(col_idx / 1.5)
            stretched_img[row_idx, col_idx] = img[row_idx, source_col]
    
    return stretched_img

def distort_img(img: np.ndarray) -> np.ndarray:
    img_height, img_width = img.shape
    center_x, center_y = img_width / 2, img_height / 2
    distortion_strength = 0.1 
    distorted_img = np.zeros_like(img)
    
    for row_idx in range(img_height):
        for col_idx in range(img_width):
            normalized_x = (col_idx - center_x) / center_x
            normalized_y = (row_idx - center_y) / center_y
            radius = np.sqrt(normalized_x**2 + normalized_y**2)
            distortion_factor = 1 + distortion_strength * radius**2
            
            source_x = int(center_x + (normalized_x / distortion_factor) * center_x)
            source_y = int(center_y + (normalized_y / distortion_factor) * center_y)
            
            if 0 <= source_x < img_width and 0 <= source_y < img_height:
                distorted_img[row_idx, col_idx] = img[source_y, source_x]
    
    return distorted_img

def rotate_img(img: np.ndarray) -> np.ndarray:
    rotated_img = np.rot90(img, -1)
    return rotated_img

def mirror_img(img: np.ndarray) -> np.ndarray:
    mirrored_img = np.fliplr(img)
    return mirrored_img

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        'translated': translate_img(img),
        'rotated': rotate_img(img),
        'stretched': stretch_img(img),
        'mirrored': mirror_img(img),
        'distorted': distort_img(img),
    }