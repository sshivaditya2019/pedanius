

# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = "./"
MODEL_DIR = "/home/sshivaditya/Projects/pedanius/saves/MultModel/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = "/home/sshivaditya/Projects/pedanius/data/train_images"
TEST_PATH = "/home/sshivaditya/Projects/pedanius/data/test_images"

# ====================================================
# CFG
# ====================================================
class CFG:
    debug = False
    num_workers = 4
    models = [
        # "tf_efficientnet_b3_ns",
        "tf_efficientnet_b4_ns",
        "vit_base_patch16_384",
        # "deit_base_patch16_384",
        "seresnext50_32x4d",
    ]
    size = {
        "tf_efficientnet_b3_ns": 512,
        "tf_efficientnet_b4_ns": 512,
        "vit_base_patch16_384": 384,
        "deit_base_patch16_384": 384,
        "seresnext50_32x4d": 512,
    }
    batch_size = 64
    seed = 7097
    target_size = 5
    target_col = "label"
    n_fold = 5
    trn_fold = {  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        "tf_efficientnet_b3_ns": {
            "best": [0, 1, 2, 3, 4],
            "final": [],
        },
        "tf_efficientnet_b4_ns": {
            "best": [0, 1, 2, 3, 4],
            "final": [],
        },
        "vit_base_patch16_384": {"best": [0, 1, 2, 3, 4], "final": []},
        "deit_base_patch16_384": {"best": [0, 1, 2, 3, 4], "final": []},
        "seresnext50_32x4d": {"best": [5, 6, 7, 8, 9], "final": []},
    }
    data_parallel = {
        "tf_efficientnet_b3_ns": False,
        "tf_efficientnet_b4_ns": True,  # True,
        "vit_base_patch16_384": False,
        "deit_base_patch16_384": False,
        "seresnext50_32x4d": False,
    }
    transform = {
        # "tf_efficientnet_b3_ns": None,
        "tf_efficientnet_b4_ns": "rotate",
        "vit_base_patch16_384": "rotate",
        # "deit_base_patch16_384": None,
        "seresnext50_32x4d": "rotate",
    }
    weight = {
        # "tf_efficientnet_b3_ns": None,
        "tf_efficientnet_b4_ns": 1,
        "vit_base_patch16_384": 1,
        # "deit_base_patch16_384": None,
        "seresnext50_32x4d": 1,
    }
    tta = 10  # 1: no TTA, >1: TTA
    no_tta_weight = tta - 1
    train = False
    inference = True

tta_weight_sum = CFG.no_tta_weight + (CFG.tta - 1)
weight_sum = sum([CFG.weight[model] for model in CFG.models]) * tta_weight_sum

# ====================================================
# Library
# ====================================================
import sys



import math
import os
import random
import shutil
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy as sp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from albumentations import (
    CenterCrop,
    CoarseDropout,
    Compose,
    Cutout,
    HorizontalFlip,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    ImageOnlyTransform,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomContrast,
    RandomCrop,
    RandomResizedCrop,
    Resize,
    Rotate,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.nn.parameter import Parameter
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file=OUTPUT_DIR + "inference.log"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_torch(seed=CFG.seed)


test = pd.read_csv("/home/sshivaditya/Projects/pedanius/data/sample_submission.csv")
test.head()


# ====================================================
# Dataset
# ====================================================
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f"{TEST_PATH}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data, size):

    if data == "train":
        return Compose(
            [
                # Resize(size, size),
                RandomResizedCrop(size, size),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                CoarseDropout(p=0.5),
                Cutout(p=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    if data == "valid":
        return Compose(
            [
                Resize(size, size),
                CenterCrop(size, size),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    if data == "simple":
        return Compose(
            [
                # Resize(size, size),
                RandomResizedCrop(size, size),
                # Transpose(p=0.5),
                # HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                # ShiftScaleRotate(p=0.5),
                # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                # CoarseDropout(p=0.5),
                # Cutout(p=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    if data == "rotate":
        return Compose(
            [
                # Resize(size, size),
                RandomResizedCrop(size, size),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                # ShiftScaleRotate(p=0.5),
                # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                # CoarseDropout(p=0.5),
                # Cutout(p=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
# ====================================================
# MODEL
# ====================================================
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", pretrained=False):
        super().__init__()

        if model_name == "deit_base_patch16_384":
            self.model = torch.hub.load("facebookresearch/deit:main", model_name, pretrained=pretrained)
            #self.model = torch.hub.load("../input/fair-deit", model_name, pretrained=pretrained, source="local")
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, CFG.target_size)

        else:
            self.model = timm.create_model(model_name, pretrained=pretrained)

            if "resnext50_32x4d" in model_name:
                n_features = self.model.fc.in_features
                self.model.fc = nn.Linear(n_features, CFG.target_size)

            elif model_name.startswith("tf_efficientnet"):
                n_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(n_features, CFG.target_size)

            elif model_name.startswith("vit_"):
                n_features = self.model.head.in_features
                self.model.head = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x
# ====================================================
# Helper functions
# ====================================================
def inference(model, states, test_loader, device, data_parallel):
    model.to(device)

    # Use multi GPU
    if device == torch.device("cuda") and data_parallel:
        model = torch.nn.DataParallel(model)  # make parallel
        # torch.backends.cudnn.benchmark=True

    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state["model"])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to("cpu").numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs

# ====================================================
# inference
# ====================================================
predictions = None
for model_name in CFG.models:
    for i in range(CFG.tta):
        model = CassvaImgClassifier(model_name, pretrained=False)
        states = []
        for saved_model in ["best", "final"]:
            if CFG.trn_fold[model_name][saved_model] != []:
                LOGGER.info(
                    f"========== Model: {model_name}, TTA: {i}, Saved: {saved_model}, Fold: {CFG.trn_fold[model_name][saved_model]} =========="
                )
                states += [
                    torch.load(MODEL_DIR + f"{model_name}_fold{fold}_{saved_model}.pth")
                    for fold in CFG.trn_fold[model_name][saved_model]
                ]

        if i == 0:  # no TTA
            test_dataset = TestDataset(test, transform=get_transforms(data="valid", size=CFG.size[model_name]))
            tta_weight = CFG.no_tta_weight
        else:
            test_dataset = TestDataset(
                test, transform=get_transforms(data=CFG.transform[model_name], size=CFG.size[model_name])
            )
            tta_weight = 1

        test_loader = DataLoader(
            test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True
        )

        inf = inference(model, states, test_loader, device, CFG.data_parallel[model_name])
        LOGGER.info(f"Inference example: {inf[0]}")

        if predictions is None:
            predictions = inf[np.newaxis] * CFG.weight[model_name] * tta_weight
        else:
            predictions = np.append(predictions, inf[np.newaxis] * CFG.weight[model_name] * tta_weight, axis=0)

sub = np.sum(predictions, axis=0) / weight_sum
LOGGER.info(f"========== Overall ==========")
LOGGER.info(f"Submission example: {sub[0]}")


# submission
test["label"] = sub.argmax(1)
test[["image_id", "label"]].to_csv(OUTPUT_DIR + "submission.csv", index=False)
test.head()
