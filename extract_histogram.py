# In file: extract_histogram.py
import numpy as np


def extract_histogram(img, q_val=12):
    """
    Extracts a color histogram from an image based on the "base-q" method
    described in the lecture slides.

    Args:
        img (np.array): A normalized RGB image with values in [0.0, 1.0].
        q_val (int): The number of quantization levels per channel (Q).

    Returns:
        np.array: A normalized 1D histogram of length Q^3.
    """
    # Q: The level of quantization. Let's use 12 as per your other files.
    Q = q_val

    # 1. Quantize each color channel independently (as per slide 2)
    # The image 'img' is already normalized to [0,1], so we multiply by Q.
    # The result is a 3D array where each channel has integer values from 0 to Q-1.
    q_img = np.floor(img * Q)

    # Separate the channels for clarity
    r_quantized = q_img[:, :, 0]
    g_quantized = q_img[:, :, 1]
    b_quantized = q_img[:, :, 2]

    # 2. Combine the 3D coordinates into a single 1D bin index (as per slide 3)
    # This is the core formula: bin = r' * Q^2 + g' * Q^1 + b'
    bin_indices = r_quantized * (Q ** 2) + g_quantized * (Q ** 1) + b_quantized

    # 'bin_indices' is now a 2D array where each element is an integer
    # from 0 to Q^3 - 1. We need to flatten it for the histogram function.
    vals = bin_indices.flatten()

    # 3. Create a 1D histogram of these bin indices.
    # The total number of bins is Q^3.
    num_bins = Q ** 3

    # Use numpy's histogram function. It's perfect for this.
    # It counts the occurrences of each value in 'vals'.
    # We specify the number of bins and the range of possible values [0, Q^3].
    hist, _ = np.histogram(vals, bins=num_bins, range=(0, num_bins))

    # 4. Normalize the histogram so the sum of its elements is 1.
    # This is done by dividing by the total number of pixels.
    hist = hist.astype(np.float64)
    hist /= np.sum(hist)  # or hist / (img.shape[0] * img.shape[1])

    return hist