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
Collection of data functions for JAX and PyTorch integration of datasets. Creates new loaders and prepares the data for training and testing.
"""
import jax.numpy as jnp
import jax
from torch.utils.data import TensorDataset, DataLoader, default_collate
import numpy as np
from jax.tree_util import tree_map
from torchvision.datasets import MNIST
from torch import randperm


def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))


class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def to_dataloader(data, batch_size, num_classes, fits_in_memory=True):
    loader = []
    if fits_in_memory:
        for dataset in data:
            images, labels = prepare_data(
                dataset[:][0], dataset[:][1], batch_size, num_classes)
            loader.append((images, labels))
    else:
        for dataset in data:
            dataloader = NumpyLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            loader.append(dataloader)
    return loader


def reshape_perm(dataset, perm):
    data, labels = dataset
    return data.reshape(data.shape[0], data.shape[1], -1)[:, :, perm].reshape(data.shape), labels


def prepare_data(data, targets, batch_size, num_classes):
    data = jnp.array(data, dtype=jnp.float32)
    targets = jax.nn.one_hot(
        jnp.array(targets, dtype=jnp.int32), num_classes=num_classes)
    data, targets = data[:len(
        data) - len(data) % batch_size], targets[:len(targets) - len(targets) % batch_size]
    return data.reshape(-1, batch_size, *data.shape[1:]), targets.reshape(-1, batch_size, num_classes)


def prepare_data_val_split(tensor_dataset_list, rng, val_split=0.1):
    """ Prepare data for training and validation split.

    Args:
        tensor_dataset_list (list[TensorDataset]): The list of TensorDataset containing the training datasets.
        rng (jax.random.PRNGKey): The random number generator.
        val_split (float): The validation split.

    Returns:
        tuple: The training and validation dataset as TensorDataset.
    """
    train_dataset_list = []
    val_dataset_list = []
    for tensor_dataset in tensor_dataset_list:
        data, labels = tensor_dataset[:]
        num_samples = data.shape[0]
        perm = randperm(num_samples)
        data = data[perm]
        labels = labels[perm]
        split = int(num_samples * (1 - val_split))
        train_dataset_list.append(TensorDataset(data[:split], labels[:split]))
        val_dataset_list.append(TensorDataset(data[split:], labels[split:]))
    return train_dataset_list, val_dataset_list
