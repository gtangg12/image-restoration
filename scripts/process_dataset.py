import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from src.image_restoration.corruption.transforms import ApplyOceanCorruption

# Define the ApplyOceanCorruption transform
ocean_transform = ApplyOceanCorruption(Lx=64, Lz=64, size=(256, 256), device='cuda')

# Define preprocessing pipeline
resize_transform = transforms.Resize((256, 256))
to_tensor_transform = transforms.ToTensor()

# Parameters
batch_size = 16
num_workers = 4

# Create a dataset of images from the source folder
image_folder = "image_restoration_src/data"
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.JPEG'))]

# Output folders
os.makedirs("ground_truth", exist_ok=True)
os.makedirs("distorted", exist_ok=True)

# Save images
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for i, image_path in enumerate(image_paths):
    # Get the filename from the path
    filename = os.path.basename(image_path)

    # Load and preprocess the original image
    original_image = Image.open(image_path).convert("RGB")
    resized_image = resize_transform(original_image)
    ground_truth_tensor = to_tensor_transform(resized_image)

    # Apply the ocean corruption transform
    distorted_tensor = ocean_transform(ground_truth_tensor.to(device))

    # Convert tensors back to PIL images
    ground_truth_image = transforms.ToPILImage()(ground_truth_tensor)
    distorted_image = transforms.ToPILImage()(distorted_tensor.cpu())

    # Save images
    ground_truth_image.save(os.path.join("ground_truth", filename))
    distorted_image.save(os.path.join("distorted", filename))