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
Bayesian Multi-Layer Perceptron (MLP) Classifier with Layer Normalization.
"""

from customLayers import BayesianLinear
from equinox import Module
from jax.random import split
from jax.nn import relu
from jax.numpy import ravel
from jax import vmap
from equinox.nn import LayerNorm, BatchNorm


class BaseBayesianMLP(Module):
    layers: list[BayesianLinear]

    def __init__(self, key, layers=[784, 50, 10], use_bias=False, activation_fn=None, norm_fn=None, norm_params=None, **kwargs):
        keys = split(key, len(layers) - 1)
        # Start with `ravel`
        self.layers = []
        # Add `BayesianLinear` and `relu` alternately except after the last linear layer
        for i in range(len(layers) - 1):
            # Add `BayesianLinear` layer
            self.layers.append(BayesianLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=use_bias,
                key=keys[i]
            ))
            if norm_fn == LayerNorm:
                norm_params["shape"] = (layers[i + 1],)
                self.layers.append(norm_fn(**norm_params))
            elif norm_fn == BatchNorm:
                norm_params["input_size"] = layers[i + 1]
                self.layers.append(norm_fn(**norm_params))
            # Add `relu` if not the last layer
            if i < len(layers) - 2:
                self.layers.append(activation_fn)

    def __call__(self, x, state, samples, key, *, backwards=False):
        s_key, key = split(key, 2)
        samples_keys = split(s_key, samples)
        x = ravel(x)
        x = vmap(forward, in_axes=(None, None, 0))(
            x, self.layers, samples_keys)
        return x, state


def forward(x, layers, key):
    for i, layer in enumerate(layers):
        if isinstance(layer, BayesianLinear):
            l_key, key = split(key, 2)
            x = layer(x, key=l_key)
        else:   # activation function
            x = layer(x)
    return x


class BayesianMLPLayerNorm(BaseBayesianMLP):
    def __init__(self, key, layers=[784, 50, 10], use_bias=False, activation_fn=None, **kwargs):
        norm_fn = LayerNorm
        norm_params = {
            "use_weight": False,
            "use_bias": False,
        }
        super().__init__(key, layers, use_bias, activation_fn, norm_fn, norm_params, **kwargs)
