import copy
from pickletools import optimize
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
from model import data_loader, resnext, net
from tqdm import tqdm
import utils
import os
from sklearn.metrics import precision_recall_fscore_support



def train_model(model, crtierion, optimizer, scheduler, num_epochs, data_loader,dataset_sizes):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train_img","test_img"]:
            if phase == "train_img":
                model.train()
            else:
                model.eval()
        
        running_loss = 0.0
        running_corrects= 0
        lab = np.array([0,1,2,3,4])
        for inputs, labels in tqdm(data_loader[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train_img"):
                output = model(inputs)
                _, preds = torch.max(output, 1)
                loss = crtierion(output, labels)

                if phase == "train_img":
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print('Precision: {:.4f} Recall: {:.4f} FBeta-Score: {:.4f}'.format(
            #pres, recall, fbeta_score
        #))
        if phase == "train_img":
            scheduler.step()
        w = precision_recall_fscore_support(labels.data.cpu(), preds.cpu(), average='macro')
        print(w)
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc
        ))
        if phase == "test_img" and epoch_acc > best_acc:
            best_ac = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elpased = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elpased // 60, time_elpased%60
    ))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    json_path = os.path.join("configs/",'params.json')
    assert os.path.isfile(json_path)
    params = utils.Params(json_path)
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model_ft = models.resnext50_32x4d(pretrained= True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Linear(num_ftrs,5)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    data_set,sizes = data_loader.dataset_entire(params)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft,criterion, optimizer_ft, exp_lr_sch, 26,data_set,sizes)
    torch.save(model_ft.state_dict(), "saves/FirstModel")



