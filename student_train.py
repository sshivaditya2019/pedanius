from http.client import MOVED_PERMANENTLY
import random
from sched import scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from model import data_loader, densenet, net, resnext,resnet,wrn,preresnet
import timm
import utils
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import math
import time
import logging
import os 
import evaluate

def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):
    """
    Train the model on steps
    Args:
        model: (torch.nn.Module) the neural network model
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: Custom Loss Function (KL Loss)
        dataloader: (torch.data.dataloader) Custom Dataloader class
        metric: (dict) Default Accuracy
        params: (Params) hyperparameters
    """

    best_accuracy = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print("The model will be runnning on",device,"device")
    model.train()
    teacher_model.eval()

    summ = []
    loss_avg = utils.RunningAverage()
    model.to(device)
    teacher_model.to(device)
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            train_batch = Variable(train_batch.to(device))
            labels_batch = Variable(labels_batch.to(device))
            optimizer.zero_grad()
            #Compute Student Model Output
            output_batch = model(train_batch)
            #Get One Batch output from
            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
            
            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
            loss.backward()
            optimizer.step()
            if i%params.save_summary_steps == 0:
                output_batch = output_batch.cpu().data.numpy()
                labels_batch = labels_batch.cpu().data.numpy()

                summary_batch = {metric:metrics[metric](output_batch,labels_batch) for metric in metrics }
                summary_batch['loss'] = loss.data[0]

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''
    def forward(self, x):
        x = self.model(x)
        return x


def train_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                        loss_fn_kd, metrics, params, model_dir, restore_file = None):
    '''
    Train the model and evaluate each epoch
    Args:
        model: (torch.nn.Module) Student Network 
        params: (Params) Hyperparameters
        model_dir: (string) Directory with pre trained teacher model
        restore_file: (string) file .pth model to restore
    '''

    best_val_acc = 0.0

    scheduler = StepLR(optimizer, step_size=150, gamma=0.1)

    for epoch in range(params.num_epochs):
        scheduler.step()
        
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        # compute number of batches in one epoch (one full pass over the training set)
        train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader,
                metrics, params)
        val_metrics = evaluate.evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc
        
        utils.save_checkpoint({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict' : optimizer.state_dict()},
                    is_best=is_best,
                    checkpoint=model_dir)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    json_path = os.path.join("configs/",'params.json')
    assert os.path.isfile(json_path)
    params = utils.Params(json_path)

    random.seed(769)
    torch.manual_seed(769)

    utils.set_logger(os.path.join('logs/','train.log'))
    logging.info("Loading the Dataset")

    trains, _ = data_loader.dataset_entire(params)
    device = torch.device(params.device)
    train_loader = trains['train_img']
    val_loader = trains['test_img']
    
    ##dev_dl = data_loader.fetch_dataloader('dev', params)

    logging.info("- done.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = net.Net(params)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn_kd = net.loss_fn_kd
    metrics = net.metrics

    teacher_model = CassvaImgClassifier("tf_efficientnet_b4_ns",5,pretrained=True)
    teacher_model.load_state_dict(torch.load("/home/sshivaditya/Projects/pedanius/saves/CrossEntropy/tf_efficientnet_b4_ns_fold_0_9"))
    teacher_model.eval()
    logging.info("Experiment - model version: {}".format(params.model_version))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    logging.info("First, loading the teacher model and computing its outputs...")
    train_evaluate_kd(model, teacher_model, train_loader, val_loader, optimizer, loss_fn_kd,
                            metrics, params, "/home/sshivaditya/Projects/pedanius/saves")