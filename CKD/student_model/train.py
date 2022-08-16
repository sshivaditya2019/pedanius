import gc
import random
import time
from flask import Config
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import teacher_model
from teacher_model.engine import get_net as gn

from teacher_model.utils import freeze_batchnorm_stats
import neptune.new as neptune
from .dataset import get_loaders
from .optim import get_optimizer_and_scheduler
from .engine import get_device, get_net, train_one_epoch, valid_one_epoch
from . import config
from .utils import *
from .loss import get_train_criterion, get_valid_criterion, loss_fn_kd


import warnings

from student_model import utils
warnings.filterwarnings("ignore")


def run_fold(fold):
    print_fn = print if not config.USE_TPU else xm.master_print
    print_fn(f"___________________________________________________")
    print_fn(f"Training Model:              {config.NET}")
    print_fn(f"Training Fold:               {fold}")
    print_fn(f"Image Dimensions:            {config.H}x{config.W}")
    print_fn(f"Mixed Precision Training:    {config.MIXED_PRECISION_TRAIN}")
    print_fn(f"Training Batch Size:         {config.TRAIN_BATCH_SIZE}")
    print_fn(f"Validation Batch Size:       {config.VALID_BATCH_SIZE}")
    print_fn(f"Accumulate Iteration:        {config.ACCUMULATE_ITERATION}")

    global net
    train_loader, valid_loader          = get_loaders(fold)
    device                              = get_device(n=fold+1)
    net                                 = net.to(device)
    scaler                              = torch.cuda.amp.GradScaler() if not config.USE_TPU and config.MIXED_PRECISION_TRAIN else None
    loss_tr                             = loss_fn_kd
    loss_fn                             = loss_fn_kd
    optimizer, scheduler                = get_optimizer_and_scheduler(net=net, dataloader=train_loader)

    gc.collect()
    teacher_mod = gn(name=config.TEACHER_NAME, pretrained=config.PRETRAINED)
    teacher_mod.load_state_dict(torch.load(config.PATH_TO_TEACHER))
    teacher_mod.eval()
    teacher_mod.to(device)
    for epoch in range(config.MAX_EPOCHS):
        epoch_start = time.time()

        if config.DO_FREEZE_BATCH_NORM and epoch < config.FREEZE_BN_EPOCHS:
            freeze_batchnorm_stats(net)

   
        tl = train_loader
        train_one_epoch(fold, epoch, teacher_mod, net, loss_tr, optimizer, tl, device, scaler=scaler, scheduler=scheduler, schd_batch_update=config.SCHEDULER_BATCH_STEP)
        del tl
        gc.collect()
        
        vl = valid_loader
        valid_one_epoch(fold, epoch,teacher_mod, net, loss_fn, optimizer, vl, device, scaler=scaler,scheduler=None, schd_loss_update=False)
        del vl
        gc.collect()
        print_fn(f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}] Time Taken for Epoch {epoch}: {time.time() - epoch_start} seconds |')

 
        torch.save(net.state_dict(
        ), os.path.join(config.WEIGHTS_PATH, f'{config.NET}/{config.NET}_fold_{fold}_{epoch}.bin'))

    torch.save(net.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del net, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()
    print_fn(f"___________________________________________________")


def train():
    
    model = neptune.init_model_version(
        model="CAS-35MODSS",
        name=config.NET, 
        project="sshivaditya/cassava", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNjVmNmI1ZS0zN2Y4LTQzMDgtYTk1Yy03NzdjMzgzNzVjYTIifQ==", # your credentials
    )
    
    global net
    torch.cuda.empty_cache()
    for fold in [3]:
        net = get_net(name=config.NET, pretrained=config.PRETRAINED)
        run_fold(fold)
    model["model"].upload("/home/sshivaditya/Projects/CKD/generated/weights/cnn/cnn_fold_3_14.bin")


if __name__ == "__main__":
    train()
