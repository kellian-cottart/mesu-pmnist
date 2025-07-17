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
Functions used to train the model on the task, and compute the loss w.r.t the algorithm.
"""
import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def compute_uncertainty(predictions):
    # if no samples, add a dimension
    if len(predictions.shape) == 2:
        predictions = jnp.expand_dims(predictions, axis=1)
    # compute softmax (n_elements, n_samples, n_classes)
    softmax_predictions = jax.nn.softmax(predictions, axis=-1)
    # Compute aleatoric and epistemic uncertainty OUT: (n_elements, n_classes)

    def element_uncertainty(predictions):
        # predictions is of shape (n_elements, n_samples, n_classes)
        mean_prediction = jnp.mean(predictions, axis=1)
        # -torch.sum(torch.log(torch.mean(Yp, dim=0)+1e-12) * torch.mean(Yp, dim=0), dim=-1)
        predictive = - jnp.sum(jnp.log2(mean_prediction + 1e-12)
                               * mean_prediction, axis=-1)
        aleatoric = - \
            jnp.sum(jnp.mean(jnp.log2(predictions + 1e-12)
                    * predictions, axis=1), axis=-1)
        epistemic = predictive - aleatoric
        return aleatoric, epistemic
    aleatoric_uncertainty, epistemic_uncertainty = element_uncertainty(
        softmax_predictions)
    return aleatoric_uncertainty, epistemic_uncertainty


@eqx.filter_jit
def compute_roc_auc(uncertainty, uncertainty_ood):
    # Compute ROC AUC
    x = jnp.linspace(0, uncertainty_ood.max(), 1000)
    tpr = jnp.array([jnp.mean(uncertainty < threshold)
                     for threshold in x])
    fpr = jnp.array([jnp.mean(uncertainty_ood < threshold)
                     for threshold in x])
    # Compute AUC using the trapezoidal rule
    auc = jnp.trapezoid(y=tpr, x=fpr)
    return auc
