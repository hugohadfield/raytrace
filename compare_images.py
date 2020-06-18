import cv2
import numpy as np

original = cv2.imread('benchmark_reference.png')
new = cv2.imread('benchmark.png')

# cv2.imshow('original ', original)
# cv2.imshow('new ', new)
# cv2.waitKey()

abs_difference = np.abs(original - new)

max_difference = np.max(abs_difference)

print('max_difference ', max_difference)

