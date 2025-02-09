# import cv2
import numpy as np
import random as rand


def salt_pepper_noise(img):

    # Get image dimensions
    row, column = img.shape

    # SALT NOISE #

    # Random number of salt noise pixels
    num_of_pxls = rand.randint(50000, 60000)

    for i in range(num_of_pxls):

        # Randomly pick pixels to whiten
        pxl_row = rand.randint(0, row - 1)
        pxl_column = rand.randint(0, column - 1)
        img[pxl_row][pxl_column] = 255

    # PEPPER NOISE #

    # Random number of pepper noise pixels
    num_of_pxls = rand.randint(50000, 60000)

    for i in range(num_of_pxls):

        # Randomly pick pixels to blacken
        pxl_row = rand.randint(0, row - 1)
        pxl_column = rand.randint(0, column - 1)
        img[pxl_row][pxl_column] = 0

    return img


def median_filter(img):

    # Get image dimensions
    row, column = img.shape

    img_denoised = np.zeros([row, column])

    # For every pixel not in the outline of the image,
    # find its neighbours and replace it with the median

    for i in range(1, row-1):
        for j in range(1, column-1):
            window = [img[i-1, j-1],
                      img[i-1, j],
                      img[i-1, j+1],
                      img[i, j-1],
                      img[i, j],
                      img[i, j+1],
                      img[i+1, j-1],
                      img[i+1, j],
                      img[i+1, j+1]]

            window = sorted(window)
            img_denoised[i, j] = window[4]

    img_denoised = img_denoised.astype(np.uint8)

    return img_denoised
