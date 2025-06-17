"""
Custom parameter for presynaptic neurons, including synaptic release probability and strength.
@author: Kellian Cottart
"""
from jaxtyping import Array
from equinox import Module


class PresynapticParameter(Module):
    probability: Array
    strength: Array

    def __init__(self, probability: Array, strength: Array):
        self.probability = probability
        self.strength = strength
