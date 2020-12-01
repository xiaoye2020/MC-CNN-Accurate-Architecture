import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageOps

class MyDataLoader(data.Dataset):
    def __init__(self,txt_path):
        super(MyDataLoader, self).__init__()
        fh = open(txt_path, 'r')
        imgs = []
        for image_path in fh:
            image_path = image_path.rstrip()
            path_label = image_path.split(' ')
            imgs.append((path_label[0], path_label[1], path_label[2]))
        self.imgs = imgs
        self.transform = transforms.Compose([
                    transforms.ToTensor()])

    def __getitem__(self, index):
        left_image_path, right_image_path,label = self.imgs[index]
        left_image=Image.open(left_image_path).convert('L')
        right_image=Image.open(right_image_path).convert('L')
        left_image = self.transform(left_image)
        right_image = self.transform(right_image)
        label=float(label)

        return left_image,right_image,label
    def __len__(self):
        return len(self.imgs)
 