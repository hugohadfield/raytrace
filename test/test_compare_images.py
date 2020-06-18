import cv2
import numpy as np


def test_compare_images():
    original = cv2.imread('../benchmark_reference.png')
    new = cv2.imread('../benchmark.png')

    abs_difference = np.abs(original - new)

    max_difference = np.max(abs_difference)

    assert max_difference == 0
