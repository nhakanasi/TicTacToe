import torch
import torch.nn as nn
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, in_chan, row = 3, col = 3 ):
        super().__init__()
        self.size = row * col
        
        self.features = nn.Sequential(
            nn.Conv2d(in_chan, 64, 3, 1, "same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, "same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, "same"),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.policy_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.size)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.global_pool(features).view(x.size(0), -1)
        
        policy_logits = self.policy_head(pooled)
        policy = torch.softmax(policy_logits, dim=1)
        
        value = torch.tanh(self.value_head(pooled))
        
        return policy, value