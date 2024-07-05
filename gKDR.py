import torch
from torch import nn
import sys
from torch.nn.functional import pdist
from scipy.spatial.distance import squareform

# Problem: gKDR only has Kexe as a resonable parameter here.
class gKDR(nn.Module):
    def __init__(self, X, Y, K, X_scale = 1.0, Y_scale = 1.0, EPS=1E-8, SGX=None, SGY=None):
        # Assuming gKDR always on a 2-D matrix. 
        N, M = torch.shape(X)

        assert(K >= 0 and K <= M)
        assert(EPS >= 0)
        assert(SGX is None or SGX > 0.0)
        assert(SGY is None or SGY > 0.0)

        Y = torch.reshape(Y, (N,1))

        if SGX is None:
            SGX = X_scale * torch.median(pdist(X))
        if SGY is None:
            SGY = Y_scale * torch.median(pdist(Y))

        I = torch.eye(N)

        SGX2 = max(SGX*SGX, sys.float_info.min)
        SGY2 = max(SGY*SGY, sys.float_info.min)

        Kx = torch.exp(-0.5 * squareform(pdist(X, p=2)) / SGX2) 
        Ky = torch.exp(-0.5 * squareform(pdist(Y, p=2)) / SGY2) 

        regularized_Kx = Kx + N*EPS*I
        tmp = Ky * torch.linalg.inv(regularized_Kx)
        F = (tmp.T * torch.linalg.inv(regularized_Kx)).T

        Dx = torch.reshape(torch.tile(X,(N,1)), (N,N,M), order='F').copy()
        Xij = Dx - torch.transpose(Dx, (1,0,2))
        Xij = Xij / SGX2
        H = Xij * torch.tile(Kx[:,:,torch.newaxis], (1,1,M))

        R = torch.zeros((M,M), order='F')

        nabla_k = torch.reshape(H, (N,N*M), order='F')
        Fm = torch.reshape(torch.matmul(F,nabla_k), (N,N,M), order='F')

        for k in range(N):
            R = R + torch.dot(H[k,:,:].T, Fm[k,:,:])

        L, V = torch.linalg.eigh(R)

        assert(torch.allclose(V.imag, 0.0))

        idx = torch.argsort(L, 0)[::-1] # sort descending

        # record B, along with some bookkeeping parameters
        self.X_scale = X_scale
        self.Y_scale = Y_scale
        self.K = K
        self.B = V[:, idx]
    
    def forward(self, X):
        return X @ self.B[:,0:self.K]
