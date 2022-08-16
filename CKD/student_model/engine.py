
import random
from sched import scheduler
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import timm
from teacher_model.utils import AccuracyMeter, AverageLossMeter

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import math
import time
import logging
import os 
from .models.models import *
from . import config
import neptune.new as neptune
run = neptune.init(
        project="sshivaditya/cassava",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNjVmNmI1ZS0zN2Y4LTQzMDgtYTk1Yy03NzdjMzgzNzVjYTIifQ==",
        name="student"
    )
run["parameters"] = config

def get_accuracy(predictions, targets, normalize=True):
    predictions = torch.argmax(predictions, dim=1)
    return accuracy_score(targets, predictions, normalize=normalize)


def train_one_epoch(fold, epoch, teacher_model, student_model, loss_fn_kd, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False ):
    student_model.train()
    mbest_accuracy = 0.0
    print_fn = print if not config.USE_TPU else xm.master_print
    t = time.time()
    running_loss = AverageLossMeter()
    running_accuracy = AccuracyMeter()
    total_steps = len(train_loader)
    pbar = enumerate(train_loader)
    optimizer.zero_grad()
    for step, (imgs, image_labels) in pbar:
        imgs, image_labels = imgs.to(device, dtype=torch.float32), image_labels.to(device, dtype=torch.int64)
        curr_batch_size = imgs.size(0)

        if (not config.USE_TPU) and config.MIXED_PRECISION_TRAIN:
            with torch.cuda.amp.autocast():
            
                image_preds = student_model(imgs)
                with torch.no_grad():
                    output_teacher_batch = teacher_model(imgs)
                loss = loss_fn_kd(image_preds, image_labels, output_teacher_batch)
                
                running_loss.update(
                    curr_batch_avg_loss=loss.item(),batch_size=curr_batch_size
                )
                score = get_accuracy(image_preds.detach().cpu(), image_labels.detach().cpu())
                total_score = score * curr_batch_size
                
                running_accuracy.update(
                    y_pred=image_preds.detach().cpu(),
                    y_true=image_labels.detach().cpu() if not config.ONE_HOT_LABEL else torch.argmax(image_labels, 1).detach().cpu(),
                    batch_size=curr_batch_size)
            
            scaler.scale(loss).backward()
            if ((step + 1) % config.ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + (step / total_steps))

        else:

            image_preds = student_model(imgs)
            with torch.no_grad():
                    output_teacher_batch = teacher_model(imgs)
            loss = loss_fn_kd(image_preds, image_labels, output_teacher_batch)
            loss.backward()
            if ((step + 1) % config.ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                if config.USE_TPU:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + (step / total_steps))
                optimizer.zero_grad()

            running_loss.update(
                curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
            running_accuracy.update(
                y_pred=image_preds.detach().cpu(),
                y_true=image_labels.detach().cpu() if not config.ONE_HOT_LABEL else torch.argmax(image_labels, 1).detach().cpu(),
                batch_size=curr_batch_size)

        if config.USE_TPU:
            loss = xm.mesh_reduce(
                'train_loss_reduce', running_loss.avg, lambda x: sum(x) / len(x))
            acc = xm.mesh_reduce(
                'train_acc_reduce', running_accuracy.avg, lambda x: sum(x) / len(x))
        else:
            loss = running_loss.avg
            acc = running_accuracy.avg
        if ((config.LEARNING_VERBOSE and (step + 1) % config.VERBOSE_STEP == 0)) or ((step + 1) == total_steps) or ((step + 1) == 1):
            run["train_student/kd_loss"].log(loss)
            run["train_student/acc"].log(acc)
            run["train_student/LR"].log(optimizer.param_groups[0]["lr"])
            description = f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | Accuracy: {acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.8f} | Time: {time.time() - t:.4f}'
            print_fn(description, flush=True)


    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(fold, epoch, teacher_model, student_model, loss_fn_kd, optimizer, valid_loader, device, scaler, scheduler=None, schd_loss_update=False ):

    
    student_model.eval()
    print_fn = print if not config.USE_TPU else xm.master_print
    t = time.time()
    running_loss = AverageLossMeter()
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    total_steps = len(valid_loader)
    pbar = enumerate(valid_loader)

    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device, dtype=torch.float32)
        image_labels = image_labels.to(device, dtype=torch.int64)

        image_preds = student_model(imgs)
        image_preds_all += [torch.argmax(image_preds,
                                         1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy() if not config.ONE_HOT_LABEL else torch.argmax(image_labels, 1).detach().cpu().numpy()]
        with torch.no_grad():
                    output_teacher_batch = teacher_model(imgs)
        loss = loss_fn_kd(image_preds, image_labels, output_teacher_batch)
        run["val_student/loss"].log(loss)
        running_loss.update(curr_batch_avg_loss=loss.item(),
                            batch_size=image_labels.shape[0])
        sample_num += image_labels.shape[0]

        if config.USE_TPU:
            loss = xm.mesh_reduce(
                'valid_loss_reduce', running_loss.avg, lambda x: sum(x) / len(x))
        else:
            loss = running_loss.avg
        if ((config.LEARNING_VERBOSE and (step + 1) % config.VERBOSE_STEP == 0)) or ((step + 1) == len(valid_loader)) or ((step + 1) == 1):
            description = f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Validation Loss: {loss:.4f}'
            print_fn(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    if config.USE_TPU:
        acc = xm.mesh_reduce('valid_acc_reduce', accuracy_score(
            image_targets_all, image_preds_all), lambda x: sum(x) / len(x))
    else:
        acc = accuracy_score(image_targets_all, image_preds_all)
    run["val_student/acc"].log(acc)
    print_fn(
        f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}] Validation Multi-Class Accuracy = {acc:.4f}')
    

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(running_loss.avg)
        else:
            scheduler.step()

def get_net(name, pretrained=False):
    net = nets[name]()
    return net

def get_device(n):
    print_fn = print if not config.USE_TPU else xm.master_print
    if not config.PARALLEL_FOLD_TRAIN:
        n = 0

    if not config.USE_GPU and not config.USE_TPU:
        print_fn(f"Device:                      CPU")
        return torch.device('cpu')
    elif config.USE_TPU:
        print_fn(f"Device:                      TPU")
        if not config.PARALLEL_FOLD_TRAIN:
            return xm.xla_device()
        else:
            return xm.xla_device(n)
    elif config.USE_GPU:
        print_fn(
            f"Device:                      GPU ({torch.cuda.get_device_name(0)})")
        return torch.device('cuda:' + str(n))
