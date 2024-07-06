import torch
from torch import nn

class ExpFilter(nn.Module):
    def __init__(self, input_size, lag, filter_size):
        super().__init__()
        self.tau = torch.ones((filter_size))
        self.M = torch.arange(start=0, end=lag, step=1).repeat(filter_size,1)
        self.g = torch.rand(size=(input_size, filter_size))
    def forward(self, x):
        # Build convolution kernel. 
        kernel = torch.exp( - (self.M) / self.tau) # Shape in (filter_size, lag)
        synapse = self.g * kernel # Generate synapse dynamics from kernels, shape in (input, lag)
        result = nn.functional.conv1d(x, synapse, padding='same')
        return result
