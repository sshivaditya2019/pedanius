import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import accuracy_score
from torch.nn.functional import normalize

from torch.utils.data import DataLoader, Dataset

from . import config
from .transforms import *


def get_img(path):
    im_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb


def get_accuracy(predictions, targets, normalize=True):
    predictions = torch.argmax(predictions, dim=1)
    return accuracy_score(targets, predictions, normalize=normalize)


def create_dirs():
    print_fn = print if not config.USE_TPU else xm.master_print
    try:
        os.mkdir(config.WEIGHTS_PATH)
        print_fn(f"Created Folder \'{config.WEIGHTS_PATH}\'")
    except FileExistsError:
        print_fn(f"Folder \'{config.WEIGHTS_PATH}\' already exists.")
    try:
        os.mkdir(os.path.join(config.WEIGHTS_PATH, f'{config.NET}'))
        print_fn(
            f"Created Folder \'{os.path.join(config.WEIGHTS_PATH, f'{config.NET}')}\'")
    except FileExistsError:
        print_fn(
            f"Folder \'{os.path.join(config.WEIGHTS_PATH, f'{config.NET}')}\' already exists.")


class AverageLossMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count


class AccuracyMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def update(self, y_pred, y_true, batch_size=1):
        self.batch_size = batch_size
        self.count += self.batch_size
        self.score = get_accuracy(y_pred, y_true)
        total_score = self.score * self.batch_size
        self.sum += total_score

    @property
    def avg(self):
        self.avg_score = self.sum/self.count
        return self.avg_score


def freeze_batchnorm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.eval()
    except ValueError:
        print('error with BatchNorm2d or LayerNorm')
        return
