## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3) ## input image size 90x90
        self.conv2 = nn.Conv2d(32, 32, 1)
        self.conv3 = nn.Conv2d(64, 64, 3) ## 78x78
        self.conv4 = nn.Conv2d(64, 64, 2) ## input 39x39
        self.conv5 = nn.Conv2d(64, 64, 3)
        # Fully connected layers
        self.FC1 = nn.Linear(4*4*64, 136)
        # Activation layer
        self.actFunc = nn.SELU()
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # pooling layer
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.actFunc(self.conv1(x)) ## input 90x90
        x1 = self.maxpool1(x) ## 44x44
        x2 = self.actFunc(self.conv2(x1)) ## 44x44
        x = torch.cat((x1, x2), dim=1) ## 44x44
        x = self.actFunc(self.conv3(x)) ## 42x42
        x = self.maxpool1(x) ## 21x21
        x = self.actFunc(self.conv4(x)) ## 20x20
        x = self.maxpool1(x) ## 10x10
        x = self.actFunc(self.conv5(x)) ## 8x8
        x = self.maxpool1(x) ## 4x4
        x = x.view(x.size(0), -1)
        x = self.actFunc(self.FC1(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
