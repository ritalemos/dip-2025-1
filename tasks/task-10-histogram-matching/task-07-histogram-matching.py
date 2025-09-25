# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    result_image = np.zeros_like(source_img)

    for channel_index in range(3):
        source_channel_flat = source_img[..., channel_index].ravel()
        reference_channel_flat = reference_img[..., channel_index].ravel()

        source_histogram, _ = np.histogram(source_channel_flat, bins=256, range=(0, 255), density=True)
        reference_histogram, _ = np.histogram(reference_channel_flat, bins=256, range=(0, 255), density=True)

        source_cdf = np.cumsum(source_histogram)
        reference_cdf = np.cumsum(reference_histogram)

        intensity_mapping = np.zeros(256, dtype=np.uint8)
        reference_intensity = 0
        
        for source_intensity in range(256):
            while reference_intensity < 255 and reference_cdf[reference_intensity] < source_cdf[source_intensity]:
                reference_intensity += 1

            intensity_mapping[source_intensity] = reference_intensity

        transformed_channel = intensity_mapping[source_img[..., channel_index]]
        result_image[..., channel_index] = transformed_channel

    return result_image