import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import Knee_dataset
from model import UNetLext
import numpy as np
import cv2
import os

def run_inference(model_path, test_csv, save_dir="predictions"):

    # Load test dataset
    df = pd.read_csv("./data/CSVs/test.csv")
    test_dataset = Knee_dataset(df, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    model = UNetLext(input_channels=1, output_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Inference loop
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)

            outputs = model(image)
            pred = torch.sigmoid(outputs)[0, 0].cpu().numpy()

            # convert to binary mask 0/255
            pred_mask = (pred > 0.5).astype(np.uint8) * 255

            # save mask
            cv2.imwrite(f"{save_dir}/pred_{i}.png", pred_mask)

            print(f"Saved: {save_dir}/pred_{i}.png")


if __name__ == "__main__":
    run_inference(
        model_path="checkpoint_epoch_10.pth",
        test_csv="test.csv",
        save_dir="predictions" 
    )
