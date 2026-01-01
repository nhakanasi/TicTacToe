import torch
import torch.nn as nn
import numpy as np

class UpScaleCNNModel(nn.Module):
    def __init__(self, board_rows=10, board_cols=10):
        super().__init__()
        
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board_size = board_rows * board_cols
        
        # Backbone: shared convolutional layers
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate flatten size: 128 channels * board spatial dimensions
        flatten_size = 128 * board_rows * board_cols
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_rows * board_cols, 64),
            nn.ReLU(),
            nn.Linear(64, self.board_size),
            nn.Softmax(dim=1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_rows * board_cols, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value