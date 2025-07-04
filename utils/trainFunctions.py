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
Functions used to train the model on the task, and compute the loss w.r.t the algorithm.
"""
import equinox as eqx
from optimizers import *
from utils.testFunctions import *
import equinox as eqx
from jax.numpy import multiply
from jax.tree import map, structure, unflatten, leaves
from jax.lax import tanh, log
from jax.random import split, uniform
from torch.utils.data import DataLoader
import jax.numpy as jnp
from jax import vmap
from models import *

@eqx.filter_jit
def loss_fn(model, images, labels, samples=None, rng=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, si_parameters=None, init_state=None):
    """Loss function for the model. Determines the appropriate loss function based on the provided parameters.

    Args:
        model: The model to evaluate.
        images: The input images.
        labels: The corresponding labels.
        samples: The number of samples for the model if it is a Bayesian model (default: None).
        rng: The random key for Bayesian models (default: None).
        ewc_parameters: Parameters for Elastic Weight Consolidation (default: None).

    Returns:
        The computed loss and gradients.
    """
    if samples is not None:
        loss_fn_to_use = bayesian_loss_fn
        loss_args = (model, images, labels, samples, rng, init_state)
    elif isinstance(model, BasePresynapticMLP):
        loss_fn_to_use = presynaptic_loss_fn
        loss_args = (model, images, labels, rng, init_state)
    elif any(param is not None for param in [ewc_online_parameters, ewc_streaming_parameters, ewc_parameters]):
        loss_fn_to_use = ewc_loss_fn
        ewc_params = ewc_online_parameters or ewc_streaming_parameters or ewc_parameters
        loss_args = (model, images, labels, ewc_params, init_state)
    elif si_parameters is not None:
        loss_fn_to_use = si_loss_fn
        loss_args = (model, images, labels, si_parameters, init_state)
    else:
        loss_fn_to_use = deterministic_loss_fn
        loss_args = (model, images, labels, init_state)
    return loss_fn_to_use(*loss_args)


@eqx.filter_value_and_grad(has_aux=True)
def si_loss_fn(model, images, labels, si_parameters, init_state=None):
    coefficient = si_parameters["coefficient"]
    omega = si_parameters["omega"]
    old_param = si_parameters["old_param"]
    predictions, state = jax.vmap(model,
                                  axis_name="batch",
                                  in_axes=(0, None),
                                  out_axes=(0, None))(images, init_state)
    output = jax.nn.log_softmax(predictions, axis=-1) * labels
    difference_squared = map(
        lambda omega, new, old: omega * (new - old)**2,
        eqx.filter(omega, eqx.is_array),
        eqx.filter(model, eqx.is_array),
        eqx.filter(old_param, eqx.is_array)
    )
    sum_params = jnp.sum(
        jnp.array([jnp.sum(param) for param in leaves(difference_squared)]))
    si_loss = -jnp.sum(output, axis=-1).sum() + coefficient * sum_params
    return si_loss, state


@eqx.filter_value_and_grad(has_aux=True)
def ewc_loss_fn(model, images, labels, ewc_parameters, init_state=None):
    fisher = ewc_parameters["fisher"]
    old_param = ewc_parameters["old_param"]
    importance = ewc_parameters["importance"]
    predictions, state = jax.vmap(model, axis_name="batch",
                                  in_axes=(0, None), out_axes=(0, None))(images, init_state)
    output = jax.nn.log_softmax(predictions, axis=-1) * labels
    difference_squared = map(
        lambda fisher, new, old: multiply(fisher,  (new - old) ** 2),
        eqx.filter(fisher, eqx.is_array),
        eqx.filter(model, eqx.is_array),
        eqx.filter(old_param, eqx.is_array)
    )
    sum_params = jnp.sum(
        jnp.array([jnp.sum(param) for param in leaves(difference_squared)]))
    ewc_loss = -jnp.sum(output, axis=-1).sum() + importance / 2 * sum_params
    return ewc_loss, state


@eqx.filter_value_and_grad(has_aux=True)
def presynaptic_loss_fn(model, images, labels, rng, init_state=None):
    """ Loss function for Bayesian models. """
    # Same rng for all images in the batch, but different for each sample
    predictions, state = jax.vmap(partial(model, backwards=True),
                                  in_axes=(0, None, None), out_axes=(0, None))(images, init_state, rng, )
    output = jax.nn.log_softmax(predictions, axis=-1) * labels
    loss = -jnp.sum(output, axis=-1).sum()
    return loss, state

@eqx.filter_value_and_grad(has_aux=True)
def bayesian_loss_fn(model, images, labels, samples, rng, init_state=None):
    """ Loss function for Bayesian models. """
    # Same rng for all images in the batch, but different for each sample
    predictions, state = jax.vmap(partial(model, backwards=True),
                                  in_axes=(0, None, None, None), out_axes=(0, None))(images, init_state, samples, rng, )
    output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1) * labels
    loss = -jnp.sum(output, axis=-1).sum()
    return loss, state


@eqx.filter_value_and_grad(has_aux=True)
def deterministic_loss_fn(model, images, labels, init_state=None):
    """ Loss function for deterministic models. """
    predictions, state = jax.vmap(model, axis_name="batch",
                                  in_axes=(0, None), out_axes=(0, None))(images, init_state)
    output = jax.nn.log_softmax(predictions, axis=-1) * labels
    # mean reduction only on deterministic models
    loss = -jnp.sum(output, axis=-1).mean()
    return loss, state


def train_fn(model, dataset, num_classes, opt_state, optimizer, train_ck, train_samples=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, si_parameters=None, init_state=None):
    args = {
        "model": model,
        "dataset": dataset,
        "num_classes": num_classes,
        "opt_state": opt_state,
        "optimizer": optimizer,
        "train_ck": train_ck,
        "train_samples": train_samples,
        "ewc_online_parameters": ewc_online_parameters,
        "ewc_streaming_parameters": ewc_streaming_parameters,
        "ewc_parameters": ewc_parameters,
        "si_parameters": si_parameters,
        "init_state": init_state
    }
    if isinstance(dataset, DataLoader):
        return dataloader_train_fn(**args)
    else:
        return all_gpu_train_fn(**args)


def dataloader_train_fn(model, dataset, num_classes, opt_state, optimizer, train_ck, train_samples=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, si_parameters=None, init_state=None):
    losses = []
    static_state = eqx.partition(model, eqx.is_array)[1]
    for i, (images, labels) in enumerate(dataset):
        images = jnp.array(images)
        labels = jax.nn.one_hot(labels, num_classes=num_classes)
        dynamic_state = eqx.partition(model, eqx.is_array)[0]
        (loss, init_state), grads = loss_fn(
            model=model,
            images=images,
            labels=labels,
            samples=train_samples,
            rng=train_ck,
            ewc_online_parameters=ewc_online_parameters,
            ewc_streaming_parameters=ewc_streaming_parameters,
            ewc_parameters=ewc_parameters,
            si_parameters=si_parameters,
            init_state=init_state)
        # Update the model using the optimizer
        updates, opt_state = optimizer.update(
            grads, opt_state, dynamic_state)
        dynamic_state = optax.apply_updates(dynamic_state, updates)
        if si_parameters is not None:
            si_parameters["w_k"] = map(
                lambda w_k, grad, old, new: w_k - grad * (new-old),
                si_parameters["w_k"],
                grads,
                eqx.filter(model, eqx.is_array),
                dynamic_state
            )
        model = eqx.combine(dynamic_state, static_state)
        if ewc_streaming_parameters is not None:
            ewc_streaming_parameters["old_param"] = eqx.filter(
                model, eqx.is_array)
            ewc_streaming_parameters = update_ewc_streaming_parameters(
                model, images, labels, train_samples, ewc_streaming_parameters)
        losses.append(loss)
    losses = jnp.array(losses)
    return model, opt_state, losses, ewc_streaming_parameters, init_state, si_parameters


@eqx.filter_jit
def all_gpu_train_fn(model, dataset, num_classes, opt_state, optimizer, train_ck, train_samples=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, si_parameters=None, init_state=None):
    """ Train the model on the task.
    Splits the model into dynamic and static parts to allow for eqx.filter_jit compilation.
    """
    task_train_images = dataset[0]
    task_train_labels = dataset[1]
    dynamic_init_state, static_state = eqx.partition(model, eqx.is_array)

    def scan_fn(carry, data):
        dynamic_state, opt_state, ewc_streaming_parameters, init_state, si_parameters = carry
        images, labels, key = data
        # Train the model
        dynamic_state, opt_state, loss, ewc_streaming_parameters, init_state, si_parameters = batch_fn(
            dynamic_state, opt_state, key, optimizer, images, labels, init_state, train_samples, static_state, ewc_online_parameters, ewc_streaming_parameters, ewc_parameters, si_parameters)
        return (dynamic_state, opt_state, ewc_streaming_parameters, init_state, si_parameters), loss

    # Split the random key for each batch of data
    train_ck = jax.random.split(train_ck, task_train_images.shape[0])
    # Use jax.lax.scan to iterate over the batches
    (dynamic_init_state, opt_state, ewc_streaming_parameters, init_state, si_parameters), losses = jax.lax.scan(
        f=scan_fn,
        init=(dynamic_init_state, opt_state,
              ewc_streaming_parameters, init_state, si_parameters),
        xs=(task_train_images, task_train_labels, train_ck)
    )
    # Combine the dynamic and static parts of the model to recover the activation functions
    model = eqx.combine(dynamic_init_state, static_state)
    return model, opt_state, losses, ewc_streaming_parameters, init_state, si_parameters


def batch_fn(dynamic_state, opt_state, keys, optimizer, images, labels, state, samples, static_state, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, si_parameters=None):
    # Combine dynamic and static parts of the model
    model = eqx.combine(dynamic_state, static_state)
    (loss, state), grads = loss_fn(
        model, images, labels, samples, keys, ewc_online_parameters, ewc_streaming_parameters, ewc_parameters, si_parameters, state)
    # Update the model using the optimizer
    dynamic_state = eqx.partition(model, eqx.is_array)[0]
    updates, opt_state = optimizer.update(grads, opt_state, dynamic_state)
    dynamic_state = optax.apply_updates(dynamic_state, updates)
    if si_parameters is not None:
        si_parameters["w_k"] = map(
            lambda w_k, grad, old, new: w_k - grad * (new-old),
            si_parameters["w_k"],
            grads,
            eqx.filter(model, eqx.is_array),
            dynamic_state)
    if ewc_streaming_parameters is not None:
        ewc_streaming_parameters["old_param"] = eqx.filter(model, eqx.is_array)
        ewc_streaming_parameters = update_ewc_streaming_parameters(
            model, images, labels, samples, ewc_streaming_parameters)
    return dynamic_state, opt_state, loss, ewc_streaming_parameters, state, si_parameters

def update_ewc_streaming_parameters(model, images, labels, samples, ewc_streaming_parameters):
    # Compute the new gradients for the fisher information matrix
    _, fisher_grads = loss_fn(model, images, labels, samples)
    fisher = map(lambda x: x ** 2, fisher_grads)
    ewc_streaming_parameters["fisher"] = map(
        lambda old, new: ewc_streaming_parameters["downweighting"] * old + new, ewc_streaming_parameters["fisher"], fisher)
    return ewc_streaming_parameters


@eqx.filter_jit
def compute_fisher(model, dataset):
    def fisher_batch_fn(fisher, images, labels):
        # Compute the loss and gradients
        _, grads = loss_fn(model, images, labels)
        # Compute the Fisher information matrix by squaring the gradients
        return map(lambda x, y: x + y**2, fisher, grads)

    def scan_fn(fisher, data):
        images, labels = data
        return fisher_batch_fn(fisher, images, labels), data

    # Split the dataset into images and labels
    task_train_images, task_train_labels = dataset
    # Compute the new fisher information matrix
    fisher = map(
        lambda x: jnp.zeros_like(x), eqx.filter(model, eqx.is_array))
    fisher, _ = jax.lax.scan(
        f=scan_fn,
        init=(fisher),
        xs=(task_train_images, task_train_labels)
    )
    # Compute expectation of the empirical Fisher information matrix
    return map(lambda x: x / task_train_images.shape[0], fisher)
