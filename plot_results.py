import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid Qt errors

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_grid(test_csv, pred_dir="predictions", limit=10):
    df = pd.read_csv(test_csv)

    images = []

    for i in range(min(limit, len(df))):
        orig = cv2.imread(df['xrays'].iloc[i], cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(f"{pred_dir}/pred_{i}.png", cv2.IMREAD_GRAYSCALE)

        combined = np.hstack([orig, pred])
        images.append(combined)

    grid = np.vstack(images)

    plt.figure(figsize=(10, 20))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.title("Original vs Predicted (Top to Bottom)")

    # Save instead of showing
    plt.savefig("results_grid.png", bbox_inches='tight')
    print("Saved: results_grid.png")

if __name__ == "__main__":
    plot_grid("./data/CSVs/test.csv", "predictions", limit=10)
