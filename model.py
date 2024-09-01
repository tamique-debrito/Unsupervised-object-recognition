from torch import nn, flatten, Tensor
import torchvision


class BasicNet(nn.Module):
    def __init__(self, classes):
        # call the parent constructor
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2= nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.AdaptiveMaxPool2d(1)
        #
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=classes)
        # initialize our softmax classifier
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor):
        #
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        #
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        #
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.squeeze(2, 3)
        #
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        #
        output = self.logSoftmax(x)
        # return the output predictions
        return output
	
class MyResnet(torchvision.models.ResNet):
    #https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    pass