## TODO: define the convolutional neural network architecture

import torch
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
        self.conv1 = nn.Conv2d(1, 32, 5, padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 4, padding=0, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 2, padding=0, bias=False)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 1, padding=0, bias=False)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(p=0)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(6*6*512, 1024, bias=False)
        self.drop6 = nn.Dropout(p=0)
        self.bn6 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.drop7 = nn.Dropout(p=0)
        self.bn7 = nn.BatchNorm1d(1024)
        
#         self.fc3 = nn.Linear(1024, 512, bias=False)
#         self.drop8 = nn.Dropout(p=0)
#         self.bn8 = nn.BatchNorm1d(512)
        
        self.final = nn.Linear(1024, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.drop1(self.bn1(x))
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.drop2(self.bn2(x))
        x = self.pool3(F.elu(self.conv3(x)))
        x = self.drop3(self.bn3(x))
        x = self.pool4(F.elu(self.conv4(x)))
        x = self.drop4(self.bn4(x))
        x = self.pool5(F.elu(self.conv5(x)))
        x = self.drop5(self.bn5(x))
        
        x = x.view(x.size()[0], -1)
        x = self.drop6(self.bn6(F.relu(self.fc1(x))))
        x = self.drop7(self.bn7(F.relu(self.fc2(x))))
#         x = self.drop8(self.bn8(F.relu(self.fc3(x))))
        
        x = self.final(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x