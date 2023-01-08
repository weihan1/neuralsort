from torch import nn
from utils import device

class DummyNNsort(nn.Module):
    def __init__(self, size):
        super().__init__()
        
        self.layer_1 = nn.Linear(in_features=size, out_features=5) 
        self.layer_2 = nn.Linear(in_features=5, out_features=size) 
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) 



