from torch import nn
from utils import device
import torch
from config.config import get_config

class DummyNNsort(nn.Module):
    '''
    A dummy neural network that takes a list of integers and returns a sorted list of integers.
    '''
    def __init__(self, size):
        super().__init__()
        self.conv_base = nn.Sequential(
            nn.Conv1d(size, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2,padding=1),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2,padding=1),
            nn.Flatten(start_dim=1, end_dim=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),   
            nn.ReLU(),
            nn.Linear(64, size)
        )
        
         
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        # print(self.classifier(self.conv_base(x)).shape)
        return self.classifier(self.conv_base(x))




class DummyNNsort2(nn.Module):
    '''
    A dummy neural network that takes a list of integers and returns a sorted list of integers.
    '''
    def __init__(self, size):
        super().__init__()

        
        self.classifier = nn.Sequential(
            nn.Linear(size, 64),   
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, size)
        )
         
    
    def forward(self, x):
        
        return self.classifier(x)
