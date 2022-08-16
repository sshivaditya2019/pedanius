
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import config

class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        self.num_channels = config.num_channels
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(3*3*64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(10, 5) 
        self.relu = nn.ReLU()
        self.dropout_rate = config.dropout_rate

    def forward(self, s):
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 32 x 32
        #s = self.bn4(self.conv4(s))                         # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s,2)) 
        s = s.view(s.size(0),-1)             # batch_size x 4*4*num_channels*4
        s = self.relu(self.fc1(s))
        s = self.fc4(s)
        return s


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)



def accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


metrics = {
    'accuracy': accuracy,
}