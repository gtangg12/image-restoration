import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg16

# Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder: Input is 3 channels (RGB)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Input channels = 3 (RGB image)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Reduces size to half (128x128 -> 64x64)
        )
        
        # Decoder: Upsample and output 3 channels (RGB)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Upsample the image
        )
        
        # Final convolution layer to ensure the output is 3 channels (RGB)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)  # Reduces channels to 3 for RGB output
    
    def forward(self, x):
        # Pass through the encoder (downsampling)
        x1 = self.encoder(x)
        
        # Pass through the decoder (upsampling)
        x2 = self.decoder(x1)
        
        # Final output (RGB image)
        x3 = self.final_conv(x2)  # Use final convolution to reduce to 3 channels (RGB)
        return x3



# Dataset Class
class PairedDataset(Dataset):
    def __init__(self, distorted_dir, ground_truth_dir, transform=None):
        self.distorted_images = sorted(os.listdir(distorted_dir))
        self.ground_truth_images = sorted(os.listdir(ground_truth_dir))
        self.distorted_dir = distorted_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform

    def __len__(self):
        return len(self.distorted_images)

    def __getitem__(self, idx):
        distorted_path = os.path.join(self.distorted_dir, self.distorted_images[idx])
        ground_truth_path = os.path.join(self.ground_truth_dir, self.ground_truth_images[idx])
        distorted = Image.open(distorted_path).convert("RGB")
        ground_truth = Image.open(ground_truth_path).convert("RGB")
        if self.transform:
            distorted = self.transform(distorted)
            ground_truth = self.transform(ground_truth)
        return distorted, ground_truth

# Data preparation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = PairedDataset("ControlNet/distorted", "ControlNet/ground_truth", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Loss and optimizer
model = UNet().to('cuda')
criterion = nn.L1Loss()  # Use L1Loss or MSELoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(1500):
    for distorted, ground_truth in dataloader:
        distorted, ground_truth = distorted.to('cuda'), ground_truth.to('cuda')
        optimizer.zero_grad()
        output = model(distorted)
        loss = criterion(output, ground_truth)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")
    if epoch % 100 == 0:
        model_save_path = f"unet_model_{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")
        

# Save the model weights after training
model_save_path = "unet_model_1500.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")