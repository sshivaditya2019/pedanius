import random
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import ShuffleSplit
import torch
import pandas as pd
import torchvision
import cv2
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data import DataLoader
import os
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm



from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

def sampler_(dataset,t="train"):
    if t=="train":
        dataset_counts = [827, 1662, 1798, 9843, 1917]
    else:
        t = "test"
        dataset_counts = [277, 547, 589, 3266, 671]
        print("test")
    num_samples = sum(dataset_counts)
    labels = [tag for _,tag in dataset]
    n_classes = 5
    print(dataset_counts)
    class_weights = [num_samples/dataset_counts[i] for i in range(n_classes)]
    weights = [class_weights[labels[i]] for i in range(num_samples)]
    torch.save(weights, 'tensor'+t+'.pt')
    print("Weights Calculated")
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    return sampler

def weightLoad():
    train = torch.load("/home/sshivaditya/Projects/pedanius/tensortrain.pt")
    train_sampler = WeightedRandomSampler(torch.DoubleTensor(train), 16049)
    test = torch.load("/home/sshivaditya/Projects/pedanius/tensortest.pt")
    test_sampler = WeightedRandomSampler(torch.DoubleTensor(test), 5350)
    return train_sampler,test_sampler
def imageCount(dataset, n_classes):
    image_count = [0]*(n_classes)
    for img in tqdm(dataset):
        image_count[img[1]] += 1
    return image_count
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb
class CassavaImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        data = pd.read_csv(annotations_file)
        frames = [data[data.label == 3].sample(frac=0.5,random_state=10) ,data[data.label == 2],data[data.label == 1],data[data.label == 4],data[data.label == 0]]
        result = pd.concat(frames)
        self.img_labels = result
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
       

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = get_img(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(image=img)['image']
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    
def dataset_entire(params):

    if params.augmentation == "yes":
        train_transforms = Compose([
            RandomResizedCrop(params.img_size, params.img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ],p=1)
        
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dev_transformer = Compose([
            CenterCrop(params.img_size, params.img_size, p=1.),
            Resize(params.img_size, params.img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
    
    cassava_dataset = CassavaImageDataset(params.annotations_file, params.img_dir, train_transforms)
    train_size = int(0.75 * cassava_dataset.__len__())
    test_size = cassava_dataset.__len__() - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(cassava_dataset, [train_size, test_size])
    #print('Sampler Starting')
    #train_sampler = sampler_(train_dataset,"train")
    #test_sampler = sampler_(test_dataset,"test")
    #train_sampler, test_sampler = weightLoad()
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers = 2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers = 2, shuffle=True)
    
    return {"train_img":train_dataloader,"test_img":test_dataloader},{"train_img":train_size,"test_img":test_size}
def fetch_dataloader(types, params):

    if params.augmentation == "yes":
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees = 45),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    cassava_dataset = CassavaImageDataset(params.annotations_file, params.img_dir, train_transforms)
    train_size = int(0.75 * cassava_dataset.__len__())
    test_size = cassava_dataset.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(cassava_dataset, [train_size, test_size])
    train_sampler = sampler_(train_dataset)
    test_sampler = sampler_(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers = 2,sampler=train_sampler, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers = 2,sampler=test_sampler, shuffle=True)

    if types == "train":
        dl = train_dataloader
    else:
        dl = test_dataloader
    
    return dl

def fetch_subset_dataloader(types, params):
    if params.augmentation == "yes":
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees = 45),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    cassava_dataset = CassavaImageDataset(params.annotations_file, params.img_dir, train_transforms)
    train_size = int(0.75 * cassava_dataset.__len__())
    test_size = cassava_dataset.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(cassava_dataset, [train_size, test_size])
    indices = list(range(train_size))
    split = int(np.floor(params.subset_percent * train_size))
    np.random.seed(230)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])
    train_dataloader = DataLoader(train_dataset, batch_size=8,sampler=train_sampler, num_workers = 2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers = 2, shuffle=True)
    if types == "train":
        dl = train_dataloader
    else:
        dl = test_dataloader
    
    return dl
