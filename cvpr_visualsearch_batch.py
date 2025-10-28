# In file: cvpr_visualsearch_batch.py
import os
import numpy as np
import scipy.io as sio
import cv2
from cvpr_compare import cvpr_compare

# --- Configuration ---
# NEW: Define the number of bins here to match your experiment!
N_BINS_PER_CHANNEL = 12

# The folder names should now reflect the number of bins for clarity.
DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = f'global_rgb_{N_BINS_PER_CHANNEL}bins'  # e.g., 'global_rgb_32bins'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
RESULTS_OUTPUT_FOLDER = f'batch_results_{N_BINS_PER_CHANNEL}bins'  # Separate results folder

NUM_QUERIES_TO_RUN = 20

# --- Display Grid Configuration (no changes here) ---
GRID_ROWS = 4
GRID_COLS = 6
NUM_RESULTS_TO_SHOW = GRID_ROWS * GRID_COLS
CELL_WIDTH = 320
CELL_HEIGHT = 280
IMG_WIDTH_IN_CELL = 310
IMG_HEIGHT_IN_CELL = 180
HIST_VIS_HEIGHT = 40


# --- Helper Function (no changes here, it's already flexible) ---
def create_histogram_visualization(image, bins, vis_height):
    vis_width = IMG_WIDTH_IN_CELL
    hist_canvas = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    channels = cv2.split(image)
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    bin_width = int(np.floor(vis_width / bins))
    if bin_width == 0: bin_width = 1  # Prevent error if bins > width
    for i, (channel, color) in enumerate(zip(channels, colors)):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=vis_height, norm_type=cv2.NORM_MINMAX)
        for j in range(bins):
            x1, y1 = j * bin_width, vis_height
            x2, y2 = (j + 1) * bin_width - 1, vis_height - int(hist[j])
            cv2.rectangle(hist_canvas, (x1, y1), (x2, y2), color, -1)
    return hist_canvas


# --- Main Script ---
# (The rest of the script is updated to use the new config variables)

os.makedirs(RESULTS_OUTPUT_FOLDER, exist_ok=True)
print(f"Results will be saved in '{RESULTS_OUTPUT_FOLDER}' folder.")

print(f'Loading descriptors from {DESCRIPTOR_SUBFOLDER}...')
ALLFEAT, ALLFILES = [], []
descriptor_dir = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)

# Check if the descriptor directory exists
if not os.path.exists(descriptor_dir):
    print(f"\nERROR: Descriptor directory not found: '{descriptor_dir}'")
    print(f"Please run 'cvpr_computedescriptors.py' with the correct subfolder for {N_BINS_PER_CHANNEL} bins.")
    exit()  # Exit the script if descriptors are missing

for filename in os.listdir(descriptor_dir):
    if filename.endswith('.mat'):
        # ... (rest of loading logic is the same)
        img_actual_path = os.path.join(IMAGE_FOLDER, 'Images', filename.replace(".mat", ".bmp"))
        descriptor_path = os.path.join(descriptor_dir, filename)
        mat_data = sio.loadmat(descriptor_path)
        ALLFILES.append(img_actual_path)
        ALLFEAT.append(mat_data['F'][0])

ALLFEAT = np.array(ALLFEAT)
NIMG = ALLFEAT.shape[0]
print(f'Finished loading {NIMG} descriptors.')

for query_run_idx in range(NUM_QUERIES_TO_RUN):
    # ... (search logic is the same)
    print(f"\n--- Starting Query Run {query_run_idx + 1}/{NUM_QUERIES_TO_RUN} ---")
    query_idx = np.random.randint(0, NIMG)
    query_img_path = ALLFILES[query_idx]
    query_feat = ALLFEAT[query_idx]
    query_filename = os.path.basename(query_img_path)
    print(f"Selected query image: {query_filename}")
    dst = []
    for i in range(NIMG):
        candidate_feat = ALLFEAT[i]
        distance = cvpr_compare(query_feat, candidate_feat)
        dst.append((distance, i))
    dst.sort(key=lambda x: x[0])

    results_canvas = np.zeros((GRID_ROWS * CELL_HEIGHT, GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)

    for i in range(NUM_RESULTS_TO_SHOW):
        if i >= len(dst): break

        result_dist, result_idx = dst[i]
        result_img_path = ALLFILES[result_idx]
        img = cv2.imread(result_img_path)
        img_resized = cv2.resize(img, (IMG_WIDTH_IN_CELL, IMG_HEIGHT_IN_CELL))

        avg_color_per_channel = cv2.mean(img)
        avg_color_text = f"Avg RGB:({int(avg_color_per_channel[2])}, {int(avg_color_per_channel[1])}, {int(avg_color_per_channel[0])})"

        # KEY CHANGE: Pass the new bin count to the visualization function
        hist_vis = create_histogram_visualization(img, bins=N_BINS_PER_CHANNEL, vis_height=HIST_VIS_HEIGHT)

        # ... (rest of the cell assembly and saving logic is the same)
        cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
        img_y_start, img_x_start = 5, 5
        cell[img_y_start:img_y_start + IMG_HEIGHT_IN_CELL, img_x_start:img_x_start + IMG_WIDTH_IN_CELL] = img_resized
        hist_y_start = img_y_start + IMG_HEIGHT_IN_CELL + 5
        cell[hist_y_start:hist_y_start + HIST_VIS_HEIGHT, img_x_start:img_x_start + IMG_WIDTH_IN_CELL] = hist_vis
        text_y_start = hist_y_start + HIST_VIS_HEIGHT + 15
        rank_text, dist_text, filename_text = f"Rank: {i}", f"Dist: {result_dist:.4f}", os.path.basename(
            result_img_path)
        font, font_scale, font_color = cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255)
        cv2.putText(cell, rank_text, (10, text_y_start), font, font_scale, font_color, 1)
        cv2.putText(cell, dist_text, (100, text_y_start), font, font_scale, font_color, 1)
        cv2.putText(cell, avg_color_text, (10, text_y_start + 15), font, font_scale, font_color, 1)
        cv2.putText(cell, filename_text, (10, text_y_start + 30), font, font_scale, font_color, 1)
        if i == 0:
            cv2.putText(cell, "(QUERY)", (CELL_WIDTH - 80, 25), font, 0.6, (0, 255, 255), 2)
            cv2.rectangle(cell, (0, 0), (CELL_WIDTH - 1, CELL_HEIGHT - 1), (0, 255, 255), 2)
        row, col = i // GRID_COLS, i % GRID_COLS
        y_start, x_start = row * CELL_HEIGHT, col * CELL_WIDTH
        results_canvas[y_start:y_start + CELL_HEIGHT, x_start:x_start + CELL_WIDTH] = cell

    output_filename = f"result_for_query_{query_filename.replace('.bmp', '.png')}"
    output_filepath = os.path.join(RESULTS_OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_filepath, results_canvas)
    print(f"Result grid saved to: {output_filepath}")

print("\n--- Batch processing complete. ---")