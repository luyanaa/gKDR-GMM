import torch
from torch import nn

class SALT(nn.Module):
    def __init__(self, input_size, lag, filter_size):
        super().__init__()
        self.kernel = torch.nn.Parameter(torch.rand(size=(filter_size, lag)))
        self.g = torch.nn.Parameter(torch.rand(size=(input_size, filter_size)))
    def forward(self, x, Yt):
        synapse = torch.matmul(self.g, self.kernel) # Generate synapse dynamics from kernels, shape in (input, lag)
        conv_kernel = torch.zeros((x.shape[1], x.shape[1], self.kernel.shape[1])) # conv1d kernel needs to be (input, output, lag)
        conv_kernel[range(conv_kernel.shape[0]), range(conv_kernel.shape[1])] = synapse[range(synapse.shape[0])] # Building diagonal kernel 
        result = nn.functional.conv1d(torch.unsqueeze(x, 0).transpose(1,2), conv_kernel, padding='same')
        result[:, Yt, :] = x.transpose(0,1)[Yt] # Passthru Yt-2 and Yt directly to GMM
        return result
