import torch
from torch import nn


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = torch.nn.Linear(3 * 50 * 50, 1024)
        self.relu3 = torch.nn.ReLU()
        self.fc2   = torch.nn.Linear(1024, 256)
        self.relu4 = torch.nn.ReLU()
        self.fc3   = torch.nn.Linear(256, 64)
        self.relu5 = torch.nn.ReLU()
        self.fc4   = torch.nn.Linear(64, 1)

        #residual / resnet /inception net/ mobile net 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        x = self.fc4(x)
        return x
