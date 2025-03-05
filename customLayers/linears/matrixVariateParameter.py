from jaxtyping import Array
from equinox import Module


class MatrixVariateParameter(Module):
    mu: Array
    sigma_1: Array
    sigma_2: Array

    def __init__(self, mu, sigma_1, sigma_2):
        self.mu = mu
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
