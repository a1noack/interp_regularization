import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """MNIST-modified LeNet-5 model.
    """
    def __init__(self, activation=F.relu, p_drop=.5):
        super(LeNet, self).__init__()
        self.activation = activation
        self.logits = None
        self.probabilities = None
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc1_drop = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_drop = nn.Dropout(p=p_drop)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1_drop(self.activation(self.fc1(x)))
        x = self.fc2_drop(self.activation(self.fc2(x)))
        self.logits = self.fc3(x)
        self.probabilities = F.log_softmax(self.logits, dim=1)
        
        return self.logits
    
class DDNet(nn.Module):
    """CIFAR-10 DDNet. Described here: https://arxiv.org/abs/1511.04508.
    """
    def __init__(self, activation=F.relu, p_drop=.5):
        super(DDNet, self).__init__()
        self.activation = activation
        self.logits = None
        self.probabilities = None
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(128*5*5, 256)
        self.fc1_drop = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(256, 256)
        self.fc2_drop = nn.Dropout(p=p_drop)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.pool2(self.activation(self.conv4(x)))
        x = x.view(x.shape[0],-1)
        x = self.fc1_drop(self.fc1(x))
        x = self.fc2_drop(self.fc2(x))
        self.logits = self.fc3(x)
        self.probabilities = F.log_softmax(self.logits, dim=1)
        
        return self.logits
    
class SimpleCNN(nn.Module):
    """Simple CNN architecture described in TensorFlow tutorial.
    """
    def __init__(self, activation=F.relu, p_drop=.4):
        super(SimpleCNN, self).__init__()
        self.activation = activation
        self.logits = None
        self.probabilities = None
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc1_drop = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))        
        x = self.pool2(self.activation(self.conv2(x)))
        x = x.view(x.shape[0],-1)
        x = self.fc1_drop(self.activation(self.fc1(x)))
        self.logits = self.fc2(x)
        self.probabilities = F.softmax(self.logits, dim=1)
        
        return self.logits