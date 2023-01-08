from torch import nn
from utils import device
import torch

class DummyNNsort(nn.Module):
    '''
    A dummy neural network that takes a list of integers and returns a sorted list of integers.
    '''
    def __init__(self, size):
        super().__init__()
        
        self.layer_1 = nn.Linear(in_features=size, out_features=64) 
        self.layer_2 = nn.Linear(in_features=64, out_features=64) 
        self.layer_3 = nn.Linear(in_features=64, out_features=size)
        self.relu = nn.ReLU()
         
    
    def forward(self, x):
        h = self.relu(self.layer_1(x))
        h = self.relu(self.layer_2(h))
        h = self.relu(self.layer_3(h))
        return h



