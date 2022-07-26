"""
   Baseline CNN, losss function and metrics
   Also customizes knowledge distillation (KD) loss function here
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):


    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        #self.conv4 = nn.Conv2d(self.num_channels*4, self.num_channels*1, 3, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(self.num_channels)
        self.fc1 = nn.Linear(3*3*64, 10)
        self.dropout = nn.Dropout(0.5)
        #self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        #self.fc2 = nn.Linear(self.num_channels*4, self.num_channels*2)    
        #self.fcbn2 = nn.BatchNorm1d(self.num_channels*2)   
        #self.fc3 = nn.Linear(self.num_channels*2, self.num_channels)  
        #self.fcbn3 = nn.BatchNorm1d(self.num_channels)  
        self.fc4 = nn.Linear(10, 5) 
        self.relu = nn.ReLU()
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 32 x 32
        #print(s.shape)
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 16 x 16
        #print(s.shape)
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 16 x 16
        #print(s.shape)
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 8 x 8
        #print(s.shape)
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 8 x 8
        #print(s.shape)
        s = F.relu(F.max_pool2d(s,2)) 
        #print(s.shape)
        #s = self.bn4(self.conv4(s))                     # batch_size x num_channels*4 x 4 x 4
        #s = F.relu(F.max_pool2d(s,2))
        #print(s.shape)
        # flatten the output for each image
        #out = out.view(out.size(0),-1)
        s = s.view(s.size(0),-1)             # batch_size x 4*4*num_channels*4
        # apply 2 fully connected layers with dropout
        #s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
        #    p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        #print(s.shape)
        #s = F.dropout(F.relu(self.fcbn2(self.fc2(s))), 
        #    p=self.dropout_rate, training=self.training)
        #print(s.shape)
        #s = F.dropout(F.relu(self.fcbn3(self.fc3(s))), 
        #    p=self.dropout_rate, training=self.training)
        #print(s.shape)                                    # batch_size x 5
        #s = self.bn4(s)
        #print(s.shape)
        s = self.relu(self.fc1(s))
        s = self.fc4(s)
        return s


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 5 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    #print(outputs.shape)
    #print(teacher_outputs.shape)
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


metrics = {
    'accuracy': accuracy,
}