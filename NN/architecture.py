import torch
from torch import nn
import collections
import random

class MLP(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input,24),
            nn.ReLU,
            nn.Linear(24,24),
            nn.ReLU,
            nn.Linear(24,output),
            nn.Softmax()
        )
        
    def forward(self,x):
        self.layers(x)
        
        
        
class memrep:
    def __init__(self):
        self.mem = collections.deque([])
        
    def push(self,s,a,r,s1,a1):
        self.mem.append([s,a,r,s1,a1])
        
    def draw_sample(self, batch_size):
        sam = random.sample(self.mem, batch_size)
        
        return sam
    
    def chk_num_sample(self):
        
        return len(self.mem)