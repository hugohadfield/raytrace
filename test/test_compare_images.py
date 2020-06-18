import matplotlib.pyplot as plt
import numpy as np


def test_compare_images():
    original = plt.imread('../benchmark_reference.png')
    new = plt.imread('../benchmark.png')

    abs_difference = np.abs(original - new)

    max_difference = np.max(abs_difference)

    assert max_difference == 0

