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
Custom parameter for presynaptic neurons, including synaptic release probability and strength.
"""
from jaxtyping import Array
from equinox import Module


class PresynapticParameter(Module):
    probability: Array
    strength: Array

    def __init__(self, probability: Array, strength: Array):
        self.probability = probability
        self.strength = strength
