import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self, root, file_path):
        self.root = root
        f = open(file_path)
        
        self.img_paths = []
        self.labels = []

        for line in f:
            data = line.rstrip().split()
            img_path, label = data[0], data[1]
            label = int(label)
            self.img_paths.append(img_path)
            self.labels.append(label)
        
        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.img_paths)

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        image_path = self.root + self.img_paths[item]
        img = Image.open(image_path)
        
        # Preprocessing: Cropping & normalization
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Change grayscale image to RGB
        if img.mode == 'L':
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = transform(rgbimg)
        else:
            img = transform(img)
          
        return img, self.labels[item]