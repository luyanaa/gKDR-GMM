"""
This file is from https://github.com/alan-turing-institute/mogp-emulator/blob/main/mogp_emulator/DimensionReduction.py
Changes are made to remove mogp-emulator specified helper functions, and build more GPU-friendly algorithms. 

MIT License

Copyright (c) 2019 The Alan Turing Institute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""This module provides classes and utilities for performing dimension
reduction.  There is a single class :class:`gKDR` which
implements the method of Fukumizu and Leng [FL13]_, and which can be
used jointly with Gaussian process emulation as in [LG17]_.
"""

import sys
import importlib
# Check cupy availability
if importlib.util.find_spec('cupy'): 
    import cupy as np
    from cupyx.scipy.spatial.distance import cdist, pdist
else: 
    import numpy as np
    from scipy.spatial.distance import cdist, pdist

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor

# Cupy not available. 
from scipy.spatial.distance import squareform

def gram_matrix(X, k):
    """Computes the Gram matrix of `X`

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
              vectors

    :param k: The covariance function

    :returns: The gram matrix of `X` under the kernel `k`, that is,
              :math:`G_{ij} = k(X_i, X_j)`
    """
    ## note: do not use squareform(pdist(X, k)) here, since it assumes
    ## that dist(x,x) == 0, which might not be the case for an arbitrary k.
    return cdist(X, X, k)

def gram_matrix_sqexp(X, sigma2):
    r"""Computes the Gram matrix of `X` under the squared expontial kernel.
    Equivalent to, but more efficient than, calling ``gram_matrix(X,
    k_sqexp)``

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
              vectors

    :param sigma2: The variance parameter of the squared exponential kernel

    :returns: The gram matrix of `X` under the squared exponential
              kernel `k_sqexp` with variance parameter `sigma2` (:math:`=\sigma^2`), that
              is, :math:`G_{ij} = k_{sqexp}(X_i, X_j; \sigma^2)`

    """
    return np.exp(-0.5 * squareform(pdist(X, 'sqeuclidean')) / sigma2)


def median_dist(X):
    """Return the median of the pairwise (Euclidean) distances between
    each row of X
    """
    return np.median(pdist(X))


class gKDR(object):

    """Dimension reduction by the gKDR method.

    See [Fukumizu1]_, [FL13]_ and [LG17]_.

    An instance of this class is callable, with the ``__call__``
    method taking an input coordinate and mapping it to a reduced
    coordinate.
    """

    def __init__(self, X, Y, K=None, X_scale = 1.0, Y_scale = 1.0, EPS=1E-8, SGX=None, SGY=None):
        """Create a gKDR object

        Given some `M`-dimensional inputs (explanatory variables) `X`,
        and corresponding one-dimensional outputs (responses) `Y`, use
        the gKDR method to produce a reduced version of the input
        space with `K` dimensions.

        :type X: ndarray, of shape (N, M)
        :param X: `N` rows of `M` dimensional input vectors

        :type Y: ndarray, of shape (N,)
        :param Y: `N` response values

        :type K: integer
        :param K: The number of reduced dimensions to use (`0 <= K <= M`).

        :type EPS: float
        :param EPS: The regularization parameter, default `1e-08`; `EPS >= 0`

        :type X_scale: float
        :param X_scale: Optional, default `1.0`.  If SGX is None (the default), scale the
                        automatically determined value for SGX by X_scale.  Otherwise ignored.

        :type Y_scale: float
        :param Y_scale: Optional, default `1.0`.  If SGY is None (the default), scale the
                        automatically determined value for SGY by Y_scale.  Otherwise ignored.

        :type SGX: float | NoneType
        :param SGX: Optional, default `None`. The kernel parameter representing the
                    scale of variation on the input space.  If `None`, then the median distance
                    between pairs of input points (`X`) is used (as computed by
                    :func:`mogp_emulator.DimensionReduction.median_dist`).  If a float is
                    passed, then this must be positive.

        :type SGY: float | NoneType
        :param SGY: Optional, default `None`. The kernel parameter representing the
                    scale of variation on the output space.  If `None`, then the median distance
                    between pairs of output values (`Y`) is used (as computed by
                    :func:`mogp_emulator.DimensionReduction.median_dist`).  If a float is
                    passed, then this must be positive.
        """

        ## Note: see the Matlab implementation ...

        N, M = np.shape(X)

        ## default K: use the entire input space
        if K is None:
            K = M

        assert(K >= 0 and K <= M)
        assert(EPS >= 0)
        assert(SGX is None or SGX > 0.0)
        assert(SGY is None or SGY > 0.0)

        Y = np.reshape(Y, (N,1))

        if SGX is None:
            SGX = X_scale * median_dist(X)
        if SGY is None:
            SGY = Y_scale * median_dist(Y)

        I = np.eye(N)

        SGX2 = max(SGX*SGX, sys.float_info.min)
        SGY2 = max(SGY*SGY, sys.float_info.min)

        Kx = gram_matrix_sqexp(X, SGX2)
        Ky = gram_matrix_sqexp(Y, SGY2)

        regularized_Kx = Kx + N*EPS*I
        tmp = Ky * np.linalg.inv(regularized_Kx)
        F = (tmp.T * np.linalg.inv(regularized_Kx)).T

        Dx = np.reshape(np.tile(X,(N,1)), (N,N,M), order='F').copy()
        Xij = Dx - np.transpose(Dx, (1,0,2))
        Xij = Xij / SGX2
        H = Xij * np.tile(Kx[:,:,np.newaxis], (1,1,M))

        R = np.zeros((M,M), order='F')

        nabla_k = np.reshape(H, (N,N*M), order='F')
        Fm = np.reshape(np.matmul(F,nabla_k), (N,N,M), order='F')

        for k in range(N):
            R = R + np.dot(H[k,:,:].T, Fm[k,:,:])

        L, V = np.linalg.eigh(R)

        assert(np.allclose(V.imag, 0.0))

        idx = np.argsort(L, 0)[::-1] # sort descending

        # record B, along with some bookkeeping parameters
        self.X_scale = X_scale
        self.Y_scale = Y_scale
        self.K = K
        self.B = V[:, idx]


    def __call__(self, X):
        """Calling a gKDR object with a vector of N inputs returns the inputs
        mapped to the reduced space.

        :type X: ndarray, of shape `(N, M)`
        :param X: `N` coordinates (rows) in the unreduced `M`-dimensional space

        :rtype: ndarray, of shape `(N, K)`
        :returns:  `N` coordinates (rows) in the reduced `K`-dimensional space
        """
        return X @ self.B[:,0:self.K]

    def _compute_loss(self, X, Y, cross_validation_folds, *params, **kwparams):
        """Compute the L1 loss of a model (produced by calling train_model), via
        cross validation.  The model is trained on input parameters `x` that
        are first reduced via the dimension reduction procedure produced by
        calling ``gKDR(x, y, *params)``.

        :type X: ndarray, of shape (N, M)
        :param X: `N` input points with dimension `M`

        :type Y: ndarray, of shape (N,)
        :param Y: the `N` model observations, corresponding to each

        :type cross_validation_folds: integer
        :param cross_validation_folds: Use this many folds for cross-validation
                                       when tuning the parameters.

        :type params: tuple
        :param params: parameters to pass to :meth:`mogp_emulator.gKDR.__init__`

        :type kwparams: dict
        :param kwparams: keyword parameters to pass to
                         :meth:`mogp_emulator.gKDR.__init__`
        """

        ## combine input and output arrays, such that if
        ## X[i] = [1,2,3] and Y[i] = 4, then XY[i] = [1,2,3,4].
        ## That is,
        ## XY[:, -1] == Y
        ## XY[:, 0:-1] == X
        ##
        model = GaussianProcessRegressor()
        model = model.fit(self.__call__(X), Y)
        error_L1 = np.mean(np.abs(Y - model.predict(self.__call__(X))))
        return error_L1

