#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
#
# Code for "Bayesian continual learning and forgetting in neural networks"
# Djohan Bonnet, Kellian Cottart, Tifenn Hirtzlin, Tarcisius Januel, Thomas Dalgaty, Elisa Vianello, Damien Querlioz
# arXiv: 2504.13569
# Portions of the code are adapted from the Pytorch project (BDS-3-Clause)
#
# Author: Kellian Cottart <kellian.cottart@gmail.com>
# Date: 2025-07-03
"""
Presynaptic linear layer with stochastic synaptic release.
"""
from jax.numpy import shape, dot, ones, expand_dims
from typing import Literal, Union
from jaxtyping import PRNGKeyArray, Array
from math import sqrt
from equinox import Module, field
from equinox import _misc
from .presynapticParameter import PresynapticParameter
from jax.random import split, normal, uniform, bernoulli
from jax import custom_vjp
import jax

class PresynapticLinear(Module, strict=True):
    """Performs a linear transformation."""

    weight: dict[str, Array]
    bias: dict[str, Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """ Initialises the Bayesian Linear Layer

        Args:
            in_features: The input size. The input to the layer should be a vector of
                shape `(in_features,)`
            out_features: The output size. The output from the layer will be a vector
                of shape `(out_features,)`.
            use_bias: Whether to add on a bias as well.
            dtype: The dtype to use for the weight and the bias in this layer.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
                on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for GaussianParameter
                initialisation. (Keyword only argument.)
        """
        dtype = _misc.default_floating_dtype() if dtype is None else dtype
        wkey, bkey = split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / (sqrt(in_features_) * sqrt(6))
        wshape = (out_features_, in_features_)
        bshape = (out_features_,)
        self.weight = PresynapticParameter(
            probability=0.5 * ones(wshape, dtype=dtype),
            strength=uniform(wkey, wshape, minval=-lim, maxval=lim),
        )
        self.bias = PresynapticParameter(
            probability=0.5 * ones(bshape, dtype=dtype) if use_bias else None,
            strength=uniform(bkey, bshape, minval=-lim, maxval=lim) if use_bias else None,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array, *, key: PRNGKeyArray = None) -> Array:
        subkey, key = split(key, 2)
        output = dot_presynaptic(x, self.weight.probability, self.weight.strength, subkey)
        if self.use_bias:
            biases = self.bias.probability * self.bias.strength
            output = output + biases
        return output

@custom_vjp
def dot_presynaptic(x, probability, strength, key):
    """Presynaptic dot product surrogate function
    Args:
        x: Input tensor of shape (batch_size, in_features)
        probability: Presynaptic probabilities of shape (out_features, in_features) 
        strength: Strength of the synapses of shape (out_features, in_features)
        key: JAX PRNG key for randomness
    Returns:
        Output tensor of shape (batch_size, out_features)
    """
    stochastic_release = bernoulli(key, p=probability)
    weight = stochastic_release * strength
    return dot(x, weight.T)

def dot_presynaptic_fwd(x, probability, strength, key):
    """ Forward pass surrogate of the dot_presynaptic function
    """
    output = dot_presynaptic(x, probability, strength, key)
    return output, (x, probability, strength, key)

def dot_presynaptic_bwd(res, grad_output):
    """ Backward pass surrogate of the dot_presynaptic function
    """
    x, probability, strength, key = res
    stochastic_release = bernoulli(key, p=probability)
    weight = stochastic_release * strength 
    return (
        dot(grad_output, weight),  # Gradient w.r.t. tensor_input
        expand_dims(grad_output, 1) @ expand_dims(x, 0),  # Gradient w.r.t. presynaptic
        expand_dims(grad_output, 1) @ expand_dims(x, 0)*probability,  # Gradient w.r.t. strength
        None  # No gradient w.r.t. key
    )
# Register the custom forward and backward functions
dot_presynaptic.defvjp(dot_presynaptic_fwd, dot_presynaptic_bwd)
