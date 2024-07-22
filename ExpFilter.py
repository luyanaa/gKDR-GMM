import torch
from torch import nn

class ExpFilter(nn.Module):
    def __init__(self, input_size, lag, filter_size):
        super().__init__()
        self.M = torch.arange(start=0, end=lag, step=1).repeat(filter_size, 1).requires_grad_(False)
        self.tau = torch.nn.Parameter(torch.ones(size=(filter_size, lag)))
        self.g = torch.nn.Parameter(torch.rand(size=(input_size, filter_size)))
    def forward(self, x, Yt):
        # Build convolution kernel
        # x = torch.unsqueeze(x, 0) # (minibatch=1, input_channel, length)
        exp_kernel = torch.exp( - self.M / self.tau) # Shape in (filter_size, lag)
        synapse = torch.matmul(self.g, exp_kernel) # Generate synapse dynamics from kernels, shape in (input, lag)
        conv_kernel = torch.zeros((x.shape[1], x.shape[1], exp_kernel.shape[1])) # Kernel in (input, output, lag)
        conv_kernel[range(conv_kernel.shape[0]), range(conv_kernel.shape[1])] = synapse[range(synapse.shape[0])]
        result = nn.functional.conv1d(torch.unsqueeze(x, 0).transpose(1,2), conv_kernel, padding='same')
        print(result.shape, x.shape)
        result[:, Yt, :] = x.transpose(0,1)[Yt] # Passthru Yt-2 and Yt directly to GMM
        return result
