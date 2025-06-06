from jaxtyping import Array
from equinox import Module


class PresynapticParameter(Module):
    presynaptic: Array
    strength: Array

    def __init__(self, presynaptic: Array, strength: Array):
        self.presynaptic = presynaptic
        self.strength = strength
