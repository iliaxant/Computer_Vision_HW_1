import cv2
# import numpy as np
from functions import *

filename = '5.png'
img = cv2.imread(filename)
img_grey = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# # #                ORIGINAL IMAGE                # # #

print('')
print('============ Original Image ============')

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.imshow('Original', img_grey)
cv2.waitKey(0)

# Transform the greyscale image into binary image by thresholding
ret, img_binary = cv2.threshold(img_grey, 205, 255, cv2.THRESH_BINARY_INV)

cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
cv2.imshow('Binary', img_binary)
cv2.waitKey(0)

# Opening to remove outline of window

strel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_opened = cv2.morphologyEx(img_binary, cv2.MORPH_ERODE, strel)
img_opened = cv2.morphologyEx(img_opened, cv2.MORPH_DILATE, strel)

cv2.namedWindow('Opened', cv2.WINDOW_NORMAL)
cv2.imshow('Opened', img_opened)
cv2.waitKey(0)

# # INTEGRAL IMAGE
#
# row, column = img_grey.shape
# img_integral = np.zeros([row + 1, column + 1], dtype="uint64")
# img_integral[1: row + 1, 1: column + 1] = img_grey
# img_integral[0, :] = np.cumsum(img_integral[0, :])
# img_integral[:, 0] = np.cumsum(img_integral[:, 0])
# img_integral = img_integral.astype(np.uint8)
#
# for i in range(1, row + 1):
#     for j in range(1, column + 1):
#         img_integral[i, j] = img_integral[i, j] - img_integral[i-1, j-1] + img_integral[i-1, j] + img_integral[i, j-1]


# LINE DETECTION #

# Dilating to form a single component in each line

strel = np.ones([30, 190], dtype="uint8")
img_dilated = cv2.morphologyEx(img_opened, cv2.MORPH_DILATE, strel)

cv2.namedWindow('Dilated - Lines', cv2.WINDOW_NORMAL)
cv2.imshow('Dilated - Lines', img_dilated)
cv2.waitKey(0)

# Find the amount of connected components
(total_lines, line_ids) = cv2.connectedComponents(img_dilated, 8, cv2.CV_32S)

print('')
print('The total number of lines in this document is: ', total_lines - 1)
print('')

# Drawing bounding box around each line
_, contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_bounded = img
for cntr in range(total_lines - 1):
    x, y, w, h = cv2.boundingRect(contours[total_lines - 2 - cntr])
    img_bounded = cv2.rectangle(img_bounded, (x, y), (x + w, y + h), (0, 0, 255), 10)
    img_bounded = cv2.putText(img_bounded, str(cntr + 1), (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                              3, (0, 0, 255), 10, cv2.LINE_AA)

    # WORD DETECTION #

    img_line = img_opened[y:(y + h), x:(x + w)]

    # Dilating to form a single component for each word

    strel = np.ones([80, 40], dtype="uint8")
    img_line_dilated = cv2.morphologyEx(img_line, cv2.MORPH_DILATE, strel)

    words = cv2.connectedComponents(img_line_dilated, 8, cv2.CV_32S)
    (total_words, word_ids) = words

    # White pixel counter for area calculation

    row, column = img_line.shape
    area = 0
    for i in range(row):
        for j in range(column):
            if img_line[i, j] == 255:
                area += 1

    # # Mean gray-level value in bounding box
    #
    # median_gray = img_integral[row, column] + img_integral[0, 0] - img_integral[row, 0] - img_integral[0, column]

    # Print line specs

    print('========= Region', cntr + 1, '=========')
    print('Area (px): ', area)
    print('Bounding Box Area(px): ', h * w)
    print('Number of Words: ', total_words - 1)
    # print('Mean gray-level value in bounding box: ', median_gray)
    print('')


cv2.namedWindow('Bounded Lines', cv2.WINDOW_NORMAL)
cv2.imshow('Bounded Lines', img_bounded)
cv2.waitKey(0)

# Save image with bounded lines
cv2.imwrite('5_bounded_og.png', img_bounded)

cv2.destroyAllWindows()

# # #                  NOISY IMAGE                  # # #

print('')
print('============ Noisy Image ============')

# Insert Salt and Pepper noise in image
img_noise = salt_pepper_noise(img_grey)

cv2.namedWindow('Salt and Pepper', cv2.WINDOW_NORMAL)
cv2.imshow('Salt and Pepper', img_noise)
cv2.waitKey(0)

# Denoise image using Median Filter
# (This step might take a few seconds to load)

print('')
print('Filtering...')
img_denoised = median_filter(img_noise)

cv2.namedWindow('Median Filter', cv2.WINDOW_NORMAL)
cv2.imshow('Median Filter', img_denoised)
cv2.waitKey(0)

# Transform the greyscale image into binary image by thresholding
ret, img_binary = cv2.threshold(img_denoised, 205, 255, cv2.THRESH_BINARY_INV)

cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
cv2.imshow('Binary', img_binary)
cv2.waitKey(0)

# Opening to remove outline of window

strel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_opened = cv2.morphologyEx(img_binary, cv2.MORPH_ERODE, strel)
img_opened = cv2.morphologyEx(img_opened, cv2.MORPH_DILATE, strel)

cv2.namedWindow('Opened', cv2.WINDOW_NORMAL)
cv2.imshow('Opened', img_opened)
cv2.waitKey(0)

# # INTEGRAL IMAGE
#
# row, column = img_denoised.shape
# img_integral[1: row + 1, 1: column + 1] = img_denoised
# img_integral[0, :] = np.cumsum(img_integral[0, :])
# img_integral[:, 0] = np.cumsum(img_integral[:, 0])
# img_integral = img_integral.astype(np.uint8)
#
# for i in range(1, row + 1):
#     for j in range(1, column + 1):
#         img_integral[i, j] = img_integral[i, j] - img_integral[i-1, j-1] + img_integral[i-1, j] + img_integral[i, j-1]

# LINE DETECTION #

# Dilating to form a single component in each line

strel = np.ones([30, 190], dtype="uint8")
img_dilated = cv2.morphologyEx(img_opened, cv2.MORPH_DILATE, strel)

cv2.namedWindow('Dilated - Lines', cv2.WINDOW_NORMAL)
cv2.imshow('Dilated - Lines', img_dilated)
cv2.waitKey(0)

# Find the amount of connected components
(total_lines, line_ids) = cv2.connectedComponents(img_dilated, 8, cv2.CV_32S)

print('')
print('The total number of lines in this document is: ', total_lines - 1)
print('')

# Drawing bounding box around each line
_, contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_bounded = img_noise
for cntr in range(total_lines - 1):
    x, y, w, h = cv2.boundingRect(contours[total_lines - 2 - cntr])
    img_bounded = cv2.rectangle(img_bounded, (x, y), (x + w, y + h),  0, 10)
    img_bounded = cv2.putText(img_bounded, str(cntr + 1), (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                              3, 0, 10, cv2.LINE_AA)

    # WORD DETECTION #

    img_line = img_opened[y:(y + h), x:(x + w)]

    # Dilating to form a single component for each word

    strel = np.ones([80, 40], dtype="uint8")
    img_line_dilated = cv2.morphologyEx(img_line, cv2.MORPH_DILATE, strel)

    words = cv2.connectedComponents(img_line_dilated, 8, cv2.CV_32S)
    (total_words, word_ids) = words

    # White pixel counter for area calculation

    row, column = img_line.shape
    area = 0
    for i in range(row):
        for j in range(column):
            if img_line[i, j] == 255:
                area += 1

    # # Mean gray-level value in bounding box
    #
    # median_gray = img_integral[row, column] + img_integral[0, 0] - img_integral[row, 0] - img_integral[0, column]

    # Print line specs

    print('========= Region', cntr + 1, '=========')
    print('Area (px): ', area)
    print('Bounding Box Area(px): ', h * w)
    print('Number of Words: ', total_words - 1)
    # print('Mean gray-level value in bounding box: ', median_gray)
    print('')


cv2.namedWindow('Bounded Lines', cv2.WINDOW_NORMAL)
cv2.imshow('Bounded Lines', img_bounded)
cv2.waitKey(0)

# Save image with bounded lines
cv2.imwrite('5_bounded_noise.png', img_bounded)

cv2.destroyAllWindows()
