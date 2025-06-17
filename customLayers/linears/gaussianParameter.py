"""
Custom parameter for both mean and variance of a Gaussian distribution.
@author: Kellian Cottart
"""
from jaxtyping import Array
from equinox import Module


class GaussianParameter(Module):
    mu: Array
    sigma: Array

    def __init__(self, mu: Array, sigma: Array):
        self.mu = mu
        self.sigma = sigma
