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
Classes and functions loading all different datasets.
"""
import jax
from jax.numpy import expand_dims, ndarray, array
import equinox as eqx
from functools import partial
from jax.tree import map
from utils.uncertaintyFunctions import compute_uncertainty
from models import *

def main_test_fn(test_dataset, num_classes, test_samples, test_ck, model, model_state, norm_params, is_permuted=False, max_permutations=None, permutations=None, fits_in_memory=False):
    if is_permuted == True:
        # Specific test function only for permuted mnist to increase speed.
        # Instead of testing each permutation separately as if they were distinct tasks, we permute at the batch level to increase speed.
        images, labels = test_dataset[0]
        accuracies, aleatoric_u, epistemic_u = test_fn_permuted_mnist(
            model=model,
            images=images,
            labels=labels,
            rng=test_ck,
            state=model_state,
            max_parallel_permutation=max_permutations,
            permutations=permutations,
            test_samples=test_samples,
            norm_params=norm_params,
        )
    elif fits_in_memory == True:
        accuracies = []
        aleatoric_u = []
        epistemic_u = []
        for task_id, task_dataset in enumerate(test_dataset):
            if norm_params is not None:
                model = model.load_tree_norm(
                    map(lambda x: x[task_id], norm_params))
            images = task_dataset[0]
            labels = task_dataset[1]
            acc, pred = test_fn_memory(model=model,
                                       images=images,
                                       labels=labels,
                                       rng=test_ck,
                                       state=model_state,
                                       test_samples=test_samples)

            aleatoric, epistemic = compute_uncertainty(pred)
            accuracies.append(acc)
            aleatoric_u.append(aleatoric)
            epistemic_u.append(epistemic)
        accuracies = array(accuracies)
        aleatoric_u = array(aleatoric_u)
        epistemic_u = array(epistemic_u)
    else:
        accuracies = []
        predictions = []
        for task_id, task_dataset in enumerate(test_dataset):
            if norm_params is not None:
                model = model.load_tree_norm(
                    map(lambda x: x[task_id], norm_params))
            task_accuracies, task_predictions = [], []
            for images, labels in task_dataset:
                images = array(images)
                labels = jax.nn.one_hot(labels, num_classes=num_classes)
                acc, pred = compute_accuracy(
                    model=model,
                    images=images,
                    labels=labels,
                    state=model_state,
                    samples=test_samples,
                    rng=test_ck)
                task_accuracies.append(acc)
                task_predictions.append(pred)
            accuracies.append(array(task_accuracies).mean())
            predictions.append(array(task_predictions))
        accuracies = array(accuracies)
        predictions = array(predictions)
        predictions = predictions.reshape(
            predictions.shape[0], predictions.shape[1]*predictions.shape[2], *predictions.shape[3:])
    return accuracies, aleatoric_u, epistemic_u


def test_fn_bayesian(model, images, state, samples, rng):
    # model = eqx.nn.inference_mode(model)
    return jax.vmap(model, axis_name="batch",  in_axes=(0, None, None, None), out_axes=(0, None))(images, state, samples, rng)


def test_fn_deterministic(model, images, state):
    # model = eqx.nn.inference_mode(model)
    return jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(images, state)

def test_fn_presynaptic(model, images, state, rng):
    return jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(images, state, rng)
    return jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(images, state, rng)

@eqx.filter_jit
def compute_accuracy(model, images, labels, state, samples=None, rng=None):
    # images.shape = (batch, height, width)
    if samples is not None:
        predictions, _ = test_fn_bayesian(model, images, state, samples, rng)
        output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1)
    elif isinstance(model, BasePresynapticMLP):
        predictions, _ = test_fn_presynaptic(model, images, state, rng)
        output = jax.nn.log_softmax(predictions, axis=-1)
    else:
        predictions, _ = test_fn_deterministic(model, images, state)
        output = jax.nn.log_softmax(predictions, axis=-1)
    accuracy = (output.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
    return accuracy, predictions


@eqx.filter_jit
def test_fn_memory(model: eqx.Module,
                   images: ndarray,
                   labels: ndarray,
                   rng,
                   state,
                   test_samples=None,):
    def compute_accuracies_predictions(images, labels, test_samples, model, state, rng):
        # First, we do a scan on the number of batches
        def scan_f(carry, data):
            image, label = data
            accuracy, predictions = compute_accuracy(
                model, image, label, state, test_samples, rng)
            return carry, (accuracy, predictions)

        _, (accuracies, predictions) = jax.lax.scan(
            f=scan_f,
            init=(),
            xs=(images, labels))
        return accuracies, predictions

    accuracies, predictions = compute_accuracies_predictions(
        images, labels, test_samples, model, state, rng)
    accuracies = expand_dims(accuracies.mean(), 0)
    predictions = predictions.reshape(
        predictions.shape[0] * predictions.shape[1], *predictions.shape[2:])
    return accuracies, predictions


@eqx.filter_jit
def test_fn_permuted_mnist(model: eqx.Module,
                           images: ndarray,
                           labels: ndarray,
                           rng,
                           state,
                           max_parallel_permutation=1,
                           permutations=None,
                           test_samples=None,
                           norm_params=None):
    """
    Test function for permuted MNIST to increase speed by batching over permutations.
    Instead of testing each permutation separately, this function permutes at the batch level.
    """
    # If the number of parallel permutations is less than the total permutations, batch them
    if max_parallel_permutation < permutations.shape[0]:
        # Reshape permutations into batches
        batched_permutations = permutations.reshape(permutations.shape[0] // max_parallel_permutation,
                                                    max_parallel_permutation, *permutations.shape[1:])
        if norm_params is not None:
            # Helper function to reshape norm_params if provided
            def reshape(x):
                return x.reshape(
                    x.shape[0] // max_parallel_permutation, max_parallel_permutation, *x.shape[1:])
            norm_params = map(reshape, norm_params)
    else:
        # If all permutations fit in one batch, expand dimensions
        batched_permutations = expand_dims(permutations, 0)
        if norm_params is not None:
            norm_params = map(lambda x: expand_dims(x, 0), norm_params)

    def distribute_batches_permutations(carry, data, model):
        """
        Vmappings over permutations and computes accuracy for each batch of permutations.
        """
        (batched_permutations, batch_norm_param) = data

        def vmap_perms(permutation, batch_norm_p, model):
            # Permute images based on the current permutation
            def distribute_batches(carry, data, model):
                """
                Computes accuracy by vmapping over different permutations.
                """
                (image, label) = data
                accuracy, predictions = compute_accuracy(
                    model, image, label, state, test_samples, rng)
                # Compute aleatoric and epistemic uncertainty
                aleatoric_u, epistemic_u = compute_uncertainty(predictions)
                return carry, (accuracy, aleatoric_u, epistemic_u)
            permuted_images = images.reshape(
                images.shape[0], images.shape[1], -1)[:, :, permutation].reshape(images.shape)
            if batch_norm_p is not None:
                model = model.load_tree_norm(batch_norm_p)
            # Scan over the batch of permuted images and labels
            _, (accuracy,  aleatoric_u, epistemic_u) = jax.lax.scan(
                f=partial(distribute_batches, model=model),
                init=(),
                xs=(permuted_images, labels))

            return accuracy,  aleatoric_u, epistemic_u
        # Vmap over all permutations in the batch
        return carry, jax.vmap(vmap_perms, in_axes=(0, 0 if batch_norm_param is not None else None, None))(batched_permutations, batch_norm_param, model)
    # Scan over all batches of permutations
    _, (accuracies,  aleatoric_u, epistemic_u) = jax.lax.scan(
        f=partial(distribute_batches_permutations, model=model),
        init=(),
        xs=(batched_permutations, norm_params)
    )
    # Reshape and compute mean accuracies
    accuracies = accuracies.reshape(
        accuracies.shape[0] * accuracies.shape[1], *accuracies.shape[2:]).mean(axis=1)
    aleatoric_u = aleatoric_u.reshape(
        aleatoric_u.shape[0] * aleatoric_u.shape[1], -1)
    epistemic_u = epistemic_u.reshape(
        epistemic_u.shape[0] * epistemic_u.shape[1], -1)
    return accuracies, aleatoric_u, epistemic_u
