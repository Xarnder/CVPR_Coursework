# In file: extract_histogram.py
import numpy as np
import cv2


def extract_histogram(img):
    """
    Extracts a color histogram from an image.
    """
    n_bins_per_channel = 12

    img_8bit = (img * 255).astype(np.uint8)

    hist = cv2.calcHist([img_8bit], [0, 1, 2], None,
                        [n_bins_per_channel, n_bins_per_channel, n_bins_per_channel],
                        [0, 256, 0, 256, 0, 256])

    # Normalize the histogram so the SUM of its elements is 1.
    # We must specify NORM_L1 to achieve this.
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

    hist_flat = hist.flatten()

    return hist_flat