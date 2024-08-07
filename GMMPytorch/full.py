from typing import Iterator, List

import torch, numpy
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily
)

from .base import MixtureModel
from .base import MixtureFamily


def make_random_cov_matrix(num_dims: int, observations_per_variable: int = 10) -> numpy.ndarray:
    """
    Make random covariance matrix using observation sampling

    :param num_dims: number of variables described by covariance matrix
    :param samples_per_variable: number of observations for each variable used
        to generated covariance matrix
    :return: random covariance matrix
    """
    if num_dims == 1:
        return numpy.array([[1.0]])

    observations = numpy.random.normal(0, 1, (num_dims, observations_per_variable))
    return numpy.corrcoef(observations)

def make_random_scale_trils(num_sigmas: int, num_dims: int) -> torch.Tensor:
    """
    Make random lower triangle scale matrix. Generated by taking the The lower
    triangle of a random covariance matrix

    :param num_sigmas: number of matrices to make
    :param num_dims: covariance matrix size
    :return: random lower triangular scale matrices
    """
    return torch.tensor(numpy.array([
        numpy.tril(make_random_cov_matrix(num_dims))
        for _ in range(num_sigmas)
    ]))


class GmmFull(MixtureModel):
    """
    Gaussian mixture model with full covariance matrix expression

    :param num_components: Number of component distributions
    :param num_dims: Number of dimensions being modeled
    :param init_radius: L1 radius within which each component mean should
        be initialized, defaults to 1.0
    :param init_mus: mean values to initialize model with, defaults to None
    """
    def __init__(
        self,
        num_components: int,
        num_dims: int,
        init_radius: float = 1.0,
        init_mus: List[List[float]] = None
    ):
        super().__init__(num_components, num_dims)

        self.mus = torch.nn.Parameter(
            torch.tensor(init_mus, dtype=torch.float32)
            if init_mus is not None
            else torch.rand(num_components, num_dims).uniform_(-init_radius, init_radius)
        )
        
        # lower triangle representation of (symmetric) covariance matrix
        self.scale_tril = torch.nn.Parameter(make_random_scale_trils(num_components, num_dims))
    

    def forward(self, x: torch.Tensor):
        mixture = Categorical(logits=self.logits)
        components = MultivariateNormal(self.mus, scale_tril=self.scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        # nll_loss = -1 * mixture_model.log_prob(x).mean()

        # return nll_loss
        return mixture_model
    
    
    def constrain_parameters(self, epsilon: float = 1e-6):
        with torch.no_grad():
            for tril in self.scale_tril:
                # cholesky decomposition requires positive diagonal
                tril.diagonal().abs_()

                # diagonal cannot be too small (singularity collapse)
                tril.diagonal().clamp_min_(epsilon)
            

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.scale_tril])
    
    
    def get_covariance_matrix(self) -> torch.Tensor:
        return self.scale_tril @ self.scale_tril.mT
