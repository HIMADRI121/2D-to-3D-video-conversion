'''import torch
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from midas.model_loader import load_model  # MiDaS model loader
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose




# Define paths
csv_path = "/Users/himad/OneDrive/Desktop/2D-3D video reconstruction/MiDaS/dataset/migration_original.csv"  # CSV file containing image paths
output_path = "path/to/output/depth_frames"
os.makedirs(output_path, exist_ok=True)

# Load MiDaS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas, transform, _ = load_model("midas_v21_small", device)  # Use 'midas_v21_small' or 'midas_v21' for higher quality

# Load CSV file containing image paths
df = pd.read_csv(csv_path)

# Column name in the CSV that holds the image paths (adjust if your column name is different)
image_column = "image_path"  # Update this to match the actual column name in your CSV

# Depth estimation function
def estimate_depth(image_path, output_file):
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert to RGB and apply transformation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).to(device)

    # Perform depth estimation
    with torch.no_grad():
        depth_map = midas(input_tensor).squeeze().cpu().numpy()

    # Normalize depth map for visualization
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Save the depth map as a JPG image
    plt.imsave(output_file, depth_map_normalized, cmap="plasma")
    print(f"Depth map saved to {output_file}")

# Iterate over each row in the CSV
for index, row in df.iterrows():
    image_path = row[image_column]  # Get the image path from the specified column
    output_file = os.path.join(output_path, f"depth_{os.path.basename(image_path).replace('.png', '.jpg')}")
    estimate_depth(image_path, output_file)

'''
import torch
from midas.model_loader import load_model

# Function to load the model
def load_midas_model(model_type='midas_v21'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, non_negative=True)
    model.eval().to(device)
    return model

model = load_midas_model('midas_v21')  # Specify your model here
