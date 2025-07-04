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
Algorithm: Presynaptic stochasticity
Simon Schug, Frederik Benzing, Angelika Steger (2021) Presynaptic stochasticity improves energy efficiency and helps alleviate the stability-plasticity dilemma eLife 10:e69884
10.7554/eLife.69884
"""
import optax
from jax.numpy import where, clip, ones
from jax.tree import map
from customLayers.linears import PresynapticParameter
import jax


def discriminant(param):
    """ Discriminate between Bayesian parameters"""
    return hasattr(param, 'probability') and hasattr(param, 'strength') and param.probability is not None and param.strength is not None


def presynaptic(lr=0.01,
                p_up=0.05,
                p_down=0.05,
                p_freeze=0.01,
                p_max=1.0,
                p_min=0.25,
                g_lim=0.1,
                ) -> optax.GradientTransformation:
    """
    Optax gradient transformation for MESU.

    Args:
        lr (float): Learning rate for the update.
        p_up (float): Probability increase for presynaptic parameters.
        p_down (float): Probability decrease for presynaptic parameters.
        p_freeze (float): Threshold below which the probability is frozen.
        p_max (float): Maximum probability value.
        p_min (float): Minimum probability value.
        g_lim (float): Gradient limit for the update.

    Returns:
        optax.GradientTransformation: The MESU update rule.
    """

    def init(params):
        # Check if all parameters are bayesian i.e that the path name contains mu or sigma
        return {'step': 0, }

    def update(gradients, state, params):
        # show memory usage of params tree
        state['step'] += 1

        def update_presynaptic(param, grad):
            """ Update the parameters based on the gradients and the prior"""
            learning_rate = lr * (p_max - param.probability)
            probability_update = where(
                grad.probability > abs(g_lim), p_up, - p_down)
            frozen_mask = where(param.probability > p_freeze, 0.0, 1.0)
            frozen_mask = where(param.probability < p_max -
                                p_freeze, 0.0, frozen_mask)
            probability_next = param.probability + probability_update*frozen_mask
            # Clip the probability to be between p_min and p_max
            probability_next = clip(probability_next, p_min, p_max)
            strength_update = (param.probability / probability_next - 1) * param.strength - \
                learning_rate / (param.probability *
                                 probability_next) * grad.strength
            return PresynapticParameter(
                probability=probability_next - param.probability,
                strength=strength_update
            )
        updates = map(
            update_presynaptic,
            params,
            gradients,
            is_leaf=discriminant)
        return updates, state

    return optax.GradientTransformation(init, update)
