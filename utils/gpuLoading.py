
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
import numpy as np
import idx2numpy
from torchvision.transforms import v2
from torchvision import datasets
import os
from torch import cat, from_numpy, LongTensor, Tensor
from torch.utils.data import TensorDataset
from torch import randperm

PATH_MNIST_X_TRAIN = "datasets/MNIST/raw/train-images-idx3-ubyte"
PATH_MNIST_Y_TRAIN = "datasets/MNIST/raw/train-labels-idx1-ubyte"
PATH_MNIST_X_TEST = "datasets/MNIST/raw/t10k-images-idx3-ubyte"
PATH_MNIST_Y_TEST = "datasets/MNIST/raw/t10k-labels-idx1-ubyte"

PATH_FASHION_MNIST_X_TRAIN = "datasets/FashionMNIST/raw/train-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TRAIN = "datasets/FashionMNIST/raw/train-labels-idx1-ubyte"
PATH_FASHION_MNIST_X_TEST = "datasets/FashionMNIST/raw/t10k-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TEST = "datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte"


class GPULoading:
    """ Load local datasets on GPU using the TensorDataset

    Args:
        device (str, optional): Device to use. Defaults to "cuda:0".
    """

    def __init__(self, device="cuda:0", root="datasets", *args, **kwargs):
        self.device = device
        self.root = root
        if "test_batch_size" in kwargs:
            self.test_batch_size = kwargs["test_batch_size"]
        if "train_batch_size" in kwargs:
            self.train_batch_size = kwargs["train_batch_size"]
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def task_selection(self, task, padding=0, *args, **kwargs):
        """ Select the task to load

        Args:
            task (str): Name of the task

        Returns:
            train (TensorDataset): Training dataset
            test (TensorDataset): Testing dataset
            shape (tuple): Shape of the data
            target_size (int): Number of classes
        """
        self.padding = padding
        if "fullpmnist" in task.lower():
            train, test = self.permuted_mnist_full(
                *args, **kwargs)
        elif "mnist" in task.lower():
            train, test = self.mnist(*args, **kwargs)
        elif "fashion" in task.lower():
            train, test = self.fashion_mnist(*args, **kwargs)
        if not isinstance(train, list):
            train = [train]
        if not isinstance(test, list):
            test = [test]
        shape = train[0][0][0].shape
        target_size = len(train[0][:][1].unique())
        return train, test, shape, target_size

    def fashion_mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_FASHION_MNIST_X_TRAIN):
            datasets.FashionMNIST("datasets", download=True)
        return self.mnist_like(PATH_FASHION_MNIST_X_TRAIN, PATH_FASHION_MNIST_Y_TRAIN,
                               PATH_FASHION_MNIST_X_TEST, PATH_FASHION_MNIST_Y_TEST, *args, **kwargs)

    def mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_MNIST_X_TRAIN):
            datasets.MNIST("datasets", download=True)
        return self.mnist_like(PATH_MNIST_X_TRAIN, PATH_MNIST_Y_TRAIN,
                               PATH_MNIST_X_TEST, PATH_MNIST_Y_TEST, *args, **kwargs)

    def mnist_like(self, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
        """ Load a local dataset on GPU corresponding either to MNIST or FashionMNIST

        Args:
            batch_size (int): Batch size
            path_train_x (str): Path to the training data
            path_train_y (str): Path to the training labels
            path_test_x (str): Path to the testing data
            path_test_y (str): Path to the testing labels
        """
        # load ubyte dataset
        train_x = idx2numpy.convert_from_file(
            path_train_x).astype(np.float32)
        train_y = idx2numpy.convert_from_file(
            path_train_y).astype(np.float32)
        test_x = idx2numpy.convert_from_file(
            path_test_x).astype(np.float32)
        test_y = idx2numpy.convert_from_file(
            path_test_y).astype(np.float32)
        # Normalize and pad the data
        dataset_normalisation = kwargs["dataset_normalisation"] if "dataset_normalisation" in kwargs else "standardised"
        train_x = normalisation(
            train_x, padding=self.padding, dataset_normalisation=dataset_normalisation)
        test_x = normalisation(test_x, padding=self.padding,
                               dataset_normalisation=dataset_normalisation)
        return TensorDataset(train_x, Tensor(train_y).type(LongTensor)), TensorDataset(test_x, Tensor(test_y).type(LongTensor))

    def permuted_mnist_full(self, n_tasks=10, *args, **kwargs):
        if not os.path.exists(PATH_MNIST_X_TRAIN):
            datasets.MNIST("datasets", download=True)
        train_dataset, test_dataset = self.mnist_like(PATH_MNIST_X_TRAIN, PATH_MNIST_Y_TRAIN,
                                                      PATH_MNIST_X_TEST, PATH_MNIST_Y_TEST, *args, **kwargs)
        permutations = [randperm(784).cpu() for _ in range(n_tasks)]
        # create a dataset with n tasks all blended together
        train_x, train_y = train_dataset.data, train_dataset.targets
        test_x, test_y = test_dataset.data, test_dataset.targets
        test_data, test_labels, train_data, train_labels = [], [], [], []
        for i in range(n_tasks):
            perm = permutations[i]
            train_x_new = train_x.view(-1,
                                       784)[:, perm].view(-1, 1, 28, 28).clone()
            test_x_new = test_x.view(-1,
                                     784)[:, perm].view(-1, 1, 28, 28).clone()
            train_data.append(train_x_new)
            test_data.append(test_x_new)
            train_labels.append(train_y)
            test_labels.append(test_y)
        train_data = cat(train_data)
        test_data = cat(test_data)
        train_labels = cat(train_labels)
        test_labels = cat(test_labels)
        return train_dataset, test_dataset

def normalisation(data, dataset_normalisation="standardised", padding=0):
    """ Normalize the pixels in train_x and test_x using transform

    Args:
        train_x (np.array): Training data
        test_x (np.array): Testing data

    Returns:
        tensor, tensor: Normalized training and testing data
    """
    # Completely convert train_x and test_x to float torch tensors
    data = from_numpy(data).float() / 255
    if len(data.size()) == 3:
        data = data.unsqueeze(1)
    normalisation_options = {
        "standardised": v2.Normalize(mean=data.mean(dim=(0, 2, 3)),
                                     std=data.std(dim=(0, 2, 3))),
        "zero_mean": v2.Normalize(mean=(0.0,), std=(1.0,)),
        "min_max": v2.Lambda(lambda x: (x - x.min() / (x.max() - x.min())))
    }
    normalisation_fn = normalisation_options.get(dataset_normalisation,
                                                 v2.Normalize(mean=data.mean(dim=(0, 2, 3)),
                                                              std=data.std(dim=(0, 2, 3))))
    transform = v2.Compose([
        normalisation_fn,
        v2.Pad(padding, fill=0, padding_mode='constant'),
    ])
    return transform(data)
