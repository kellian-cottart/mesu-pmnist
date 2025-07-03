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
Multi-Layer Perceptron (MLP) Classifier with optional normalization layers.
"""
from equinox.nn import Linear, LayerNorm, BatchNorm
from equinox import Module
from jax.random import split
from jax.numpy import ravel
from customLayers import *


class BaseMLP(Module):
    layers: list[Linear]

    def __init__(self, key, layers=[784, 50, 10], use_bias=False, activation_fn=None, norm_fn=None, norm_params=None, **kwargs):
        keys = split(key, len(layers) - 1)
        self.layers = []
        # Add `BayesianLinear` and `relu` alternately except after the last linear layer
        for i in range(len(layers) - 1):
            # Add `BayesianLinear` layer
            self.layers.append(Linear(
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
            if i < len(layers) - 2 and activation_fn:
                self.layers.append(activation_fn)

    def __call__(self, x, state, *, backwards=False):
        x = ravel(x)
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return x, state


class MLPLayerNorm(BaseMLP):
    def __init__(self, key, layers=[784, 50, 10], use_bias=None, activation_fn=None, ** kwargs):
        norm_fn = LayerNorm
        norm_params = {
            "use_weight": False,
            "use_bias": False,
        }
        super().__init__(key, layers, use_bias, activation_fn, norm_fn, norm_params, **kwargs)


class MLPBatchNorm(BaseMLP):
    def __init__(self, key, layers=[784, 50, 10], use_bias=None, activation_fn=None, ** kwargs):
        norm_fn = BatchNorm
        norm_params = {
            "axis_name": "batch",
            "channelwise_affine": False,
            "momentum": 0.15,
            "eps": 1e-5,
            "inference": False,
        }
        super().__init__(key, layers, use_bias, activation_fn, norm_fn, norm_params, **kwargs)
