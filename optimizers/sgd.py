#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
#
# Code for "Bayesian continual learning and forgetting in neural networks"
# Djohan Bonnet, Kellian Cottart, Tifenn Hirtzlin, Tarcisius Januel, Thomas Dalgaty, Elisa Vianello, Damien Querlioz
# arXiv: 2504.13569
# Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
#
# Author: Kellian Cottart <kellian.cottart@gmail.com>
# Date: 2025-07-03
"""
Algorithm: Stochastic Gradient Descent (SGD)
"""
import jax
import optax
from jax.tree import map


def sgd(lr: float = 0.001,):
    """
    SGD

    Args:
        lr: Learning rate for the optimizer.
    """

    def init(params):
        return {
            'step': 0,
        }

    def update(gradients, state, params=None):
        state['step'] += 1
        # Update the parameters using sgd
        return map(lambda g: -lr * g, gradients), state

    return optax.GradientTransformation(init, update)
