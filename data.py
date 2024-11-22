import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class TextureDataset(Dataset):
    label_map = {
        'good': 0,
        'color': 1,
        'cut': 2,
        'hole': 3
        }
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
       
        image = cv2.resize(image, (28, 28))
        if image is None:
            raise ValueError(f"Error reading image {img_path}")
        
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
        label_str = img_name.split('_')[1]
        label = self.label_map[label_str]
        
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = image.float() / 255

        if self.transform:
            image = self.transform(image)

        return image, label

