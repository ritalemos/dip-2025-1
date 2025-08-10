import cv2 as cv
import numpy as np
from pathlib import Path

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    
    # TODO: Implement noise removal here (e.g., median filtering)

    total_pixels = image.size
    noisy_pixels = np.count_nonzero((image == 0) | (image == 255))
    noise_density = noisy_pixels / total_pixels

    #print(noise_density)
    if noise_density < 0.10:
        ksize = 3
    elif noise_density < 0.30:
        ksize = 5
    elif noise_density < 0.50:
        ksize = 7
    else:
        ksize = 9

    image = cv.medianBlur(image, ksize)

    return image


if __name__ == "__main__":
    noisy_image = cv.imread("noisy_image.png", cv.IMREAD_GRAYSCALE)
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv.imwrite("denoised_image.png", denoised_image)