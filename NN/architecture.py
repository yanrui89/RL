import torch
from torch import nn
import collections
from collections import OrderedDict
import random

class MLP(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,output),
            nn.Softmax()
            
        )
        
    def forward(self,x):
        self.layers(x)
        
       
class memrep:
    def __init__(self):
        self.mem = collections.deque([])
        
    def push(self,s0,s1,a,r,s10,s11,a1):
        self.mem.append([s0,s1,a,r,s10,s11,a1])
        
    def draw_sample(self, batch_size):
        sam = random.sample(self.mem, batch_size)
        
        return sam
    
    def chk_num_sample(self):
        
        return len(self.mem)

class PolicyNet(nn.Module):

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self._create_network()

    def _create_network(self):
        self.polnet = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(self.num_states, 24)),
            ('relu1', nn.ReLU()),
            ('Layer2', nn.Linear(24,24)),
            ('relu2', nn.ReLU()),
            ('Layer3', nn.Linear(24, self.num_actions)),
            ('softmax', nn.Softmax())
        ]))

    def forward(self, x):
        output = self.polnet(x)

        return output
