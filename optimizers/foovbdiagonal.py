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
Algorithm: FOO-VB Diagonal
Zeno, Chen and Golan, Itay and Hoffer, Elad and Soudry, Daniel, (2021) Task-agnostic continual learning using online variational bayes with fixed-point updates, Neural Computation
10.1162/neco_a_01430
"""

import optax
import jax.numpy as jnp
import jax
from customLayers.linears import GaussianParameter


def discriminant(param):
    """ Discriminate between Bayesian parameters"""
    return hasattr(param, 'mu') and hasattr(param, 'sigma') and param.mu is not None and param.sigma is not None


def foovbdiagonal(
        lr_mu: float = 1,
        lr_sigma: float = 1,
        clamp_grad: float = 0.) -> optax.GradientTransformation:
    """
    Optax gradient transformation for foovb-diagonal.

    Args:
        lr_mu: Learning rate for mu parameters.
        lr_sigma: Learning rate for sigma parameters.
        clamp_grad: Gradient clamping threshold.
    Returns:
        optax.GradientTransformation: The BGD update rule.
    """

    def init(params):
        # Check if all parameters are bayesian i.e that the path name contains mu or sigma
        return {
            'step': 0,
        }

    def update(gradients, state, params=None):
        state['step'] += 1
        # Update the parameters using sgd

        def update_bgd(param, grad):
            """ Update the parameters based on the gradients and the prior"""
            # If clamp_grad > 0, then clamp gradients between -clamp_grad/sigma and clamp_grad/sigma
            if clamp_grad > 0:
                grad = jax.tree_util.tree_map(
                    lambda x: jnp.clip(x, -clamp_grad / param.sigma, clamp_grad / param.sigma), grad)
            variance = param.sigma ** 2
            mu = - lr_mu * variance * grad.mu
            sigma = - lr_sigma*(0.5 * variance * grad.sigma + param.sigma *
                                (-1 + (1 + 0.25 * (variance * (grad.sigma ** 2))) ** 0.5))
            return GaussianParameter(mu, sigma)

        updates = jax.tree.map(
            update_bgd,
            params,
            gradients,
            is_leaf=discriminant
        )
        return updates, state

    return optax.GradientTransformation(init, update)
