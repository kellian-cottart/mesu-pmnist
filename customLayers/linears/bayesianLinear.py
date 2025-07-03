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
Linear layer with Bayesian weights and bias.
"""
from jax.numpy import shape, dot, ones
from typing import Literal, Union
from jaxtyping import PRNGKeyArray, Array
from math import sqrt
from equinox import Module, field
from equinox import _misc
from .gaussianParameter import GaussianParameter
from jax.random import split, normal, uniform


class BayesianLinear(Module, strict=True):
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
        lim = 4 / sqrt(in_features_)
        wshape = (out_features_, in_features_)
        bshape = (out_features_,)
        self.weight = GaussianParameter(
            mu=uniform(wkey, wshape, minval=-lim, maxval=lim),
            sigma=ones(wshape, dtype) * 0.5 * lim,
        )
        self.bias = GaussianParameter(
            mu=uniform(bkey, bshape, minval=-lim,
                       maxval=lim) if use_bias else None,
            sigma=ones(bshape, dtype) * 0.5 * lim if use_bias else None,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> Array:
        """ Call function for bayesian linear layer
        `samples` forward passes using weights reparametrization w = mu + sigma * epsilon, epsilon ~ N(0, 1)

        Args:
            x: input tensor
            samples: number of samples
            rng: random key
        """
        wkey, bkey = split(key, 2)
        weights = self.weight.mu + self.weight.sigma * \
            normal(wkey, shape(self.weight.mu))
        output = dot(weights, x)
        if self.use_bias:
            biases = self.bias.mu + self.bias.sigma * \
                normal(bkey, shape(self.bias.mu))
            output = output + biases
        return output
