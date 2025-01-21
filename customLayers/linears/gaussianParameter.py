from jaxtyping import Array
from equinox import Module


class GaussianParameter(Module):
    mu: Array
    sigma: Array

    def __init__(self, mu: Array, sigma: Array):
        self.mu = mu
        self.sigma = sigma
