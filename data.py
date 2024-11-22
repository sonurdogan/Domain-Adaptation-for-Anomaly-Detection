import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image 
import glob
from sklearn.model_selection import train_test_split


class FlawDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_FlawDataset(data_dir, transform, split_data=True):
    classes = ['cut', 'hole', 'color', 'good']
    file_paths = []
    labels = []

    for class_index, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        for img_path in glob.glob(os.path.join(class_folder, '*.png')):
            file_paths.append(img_path)
            labels.append(class_index)

    if not split_data:
        return FlawDataset(file_paths, labels, transform=transform)
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(file_paths, labels, stratify=labels, test_size=0.2, random_state=42)

    train_dataset = FlawDataset(train_paths, train_labels, transform=transform)
    test_dataset = FlawDataset(test_paths, test_labels, transform=transform)

    return train_dataset, test_dataset
