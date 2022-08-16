import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset


from . import config
from .utils import *
from .transforms import *




def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2





class MonocularDepthImageMasking:
    def __init__(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas.cuda().eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.default_transform

    def get_depth(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to("cuda")
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
            ).squeeze()
        output = prediction.cpu().numpy()
        img_min = np.min(output)
        img_max = np.max(output)
        return output > ((img_min+img_max)/3)

    def crop_image(self, image, depth):
        depth = depth.astype(int)
        mask_3d = np.stack((depth, depth, depth), axis=2)
        masked_arr = np.where(mask_3d == 1, image, mask_3d).astype(np.uint8)
        c = np.where(masked_arr != [0, 0, 0])
        x_max = max(c[1])
        x_min = min(c[1])
        y_max = max(c[0])
        y_min = min(c[0])
        image = masked_arr[y_min:y_max, x_min:x_max]
        return image

    def __call__(self, image):
        process = image
        depth = self.get_depth(process)
        image = self.crop_image(image, depth)
        return image


class CassavaDataset(Dataset):
    def __init__(self, df, data_root,
                 transforms=None,
                 output_label=True,
                 ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        self.one_hot_label = config.ONE_HOT_LABEL

        if config.DO_DEPTH_MASKING:
            self.masker = MonocularDepthImageMasking()

        if output_label == True:
            self.labels = self.df['label'].values
            # print(self.labels)

            if self.one_hot_label is True:
                self.labels = np.eye(self.df['label'].max() + 1)[self.labels]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
        else:
            target = None

        img = get_img("{}/{}".format(self.data_root,
                                     self.df.loc[index]['image_id']))

        if config.DO_DEPTH_MASKING:
            img = self.masker(img)

        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.output_label == True:
            return img, target
        else:
            return img


def get_train_dataloader(train, data_root=config.TRAIN_IMAGES_DIR):
    if config.USE_TPU:
        train_dataset = CassavaDataset(train, data_root, transforms=get_train_transforms(
        ), output_label=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),  # divide dataset among this many replicas
            rank=xm.get_ordinal(),  # which replica/device/core
            shuffle=True)
        return DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            sampler=train_sampler,
            num_workers=config.CPU_WORKERS,
            drop_last=config.DROP_LAST)
    else:
        return DataLoader(
            CassavaDataset(train, data_root, transforms=get_train_transforms(
            ), output_label=True),
            batch_size=config.TRAIN_BATCH_SIZE,
            drop_last=config.DROP_LAST,
            num_workers=config.CPU_WORKERS,
            shuffle=True)


def get_valid_dataloader(valid, data_root=config.TRAIN_IMAGES_DIR):
    if config.USE_TPU:
        valid_dataset = CassavaDataset(valid, data_root, transforms=get_valid_transforms(
        ), output_label=True)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
        return DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            sampler=valid_sampler,
            num_workers=config.CPU_WORKERS,
            drop_last=config.DROP_LAST)
    else:
        return DataLoader(
            CassavaDataset(valid, data_root, transforms=get_valid_transforms(
            ), output_label=True),
            batch_size=config.VALID_BATCH_SIZE,
            drop_last=config.DROP_LAST,
            num_workers=config.CPU_WORKERS,
            shuffle=False)


def get_infer_dataloader(infer, data_root=config.TRAIN_IMAGES_DIR):
    OK = False
    if config.USE_TPU and OK:
        infer_dataset = CassavaDataset(infer, data_root, transforms=get_valid_transforms(
        ), output_label=True)
        infer_sampler = torch.utils.data.distributed.DistributedSampler(
            infer_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
        return DataLoader(
            infer_dataset,
            batch_size=32,
            sampler=infer_sampler,
            num_workers=config.CPU_WORKERS,
            drop_last=False)
    else:
        return DataLoader(
            CassavaDataset(infer, data_root, transforms=get_valid_transforms(
            ), output_label=True),
            batch_size=32,
            drop_last=False,
            num_workers=config.CPU_WORKERS,
            shuffle=False)


def get_loaders(fold):
    train_folds = pd.read_csv(config.TRAIN_FOLDS)
    train = train_folds[train_folds.fold != fold]
    valid = train_folds[train_folds.fold == fold]

    train_loader = get_train_dataloader(
        train.drop(['fold'], axis=1))
    valid_loader = get_valid_dataloader(
        valid.drop(['fold'], axis=1))

    return train_loader, valid_loader
