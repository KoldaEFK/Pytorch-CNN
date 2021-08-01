import torch.nn as nn
import torch.nn.functional as F

#MODEL
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,3,5) #input shape batch_size*1*28*28
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(3,10,5)
        self.fc1 = nn.Linear(10*4*4, 120) #shape before flattening is batch_size*10*4*4
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,10*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
