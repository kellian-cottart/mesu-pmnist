import jax
from jax.numpy import expand_dims, ndarray, array
import equinox as eqx
import functools as ft
from jax.tree import map


def main_test_fn(test_dataset, num_classes, test_samples, test_ck, model, model_state, norm_params, is_permuted=False, max_permutations=None, permutations=None, fits_in_memory=False):
    if is_permuted == True:
        # Specific test function only for permuted mnist to increase speed.
        # Instead of testing each permutation separately as if they were distinct tasks, we permute at the batch level to increase speed.
        images, labels = test_dataset[0]
        accuracies, predictions = test_fn_permuted_mnist(
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
        predictions = []
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
            accuracies.append(acc)
            predictions.append(pred)
        accuracies = array(accuracies)
        predictions = array(predictions)
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
    return accuracies, predictions


def test_fn_bayesian(model, images, state, samples, rng):
    # model = eqx.nn.inference_mode(model)
    return jax.vmap(model, axis_name="batch",  in_axes=(0, None, None, None), out_axes=(0, None))(images, state, samples, rng)


def test_fn_deterministic(model, images, state):
    # model = eqx.nn.inference_mode(model)
    return jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(images, state)


@ eqx.filter_jit
def compute_accuracy(model, images, labels, state, samples=None, rng=None):
    # images.shape = (batch, height, width)
    if samples is not None:
        predictions, _ = test_fn_bayesian(model, images, state, samples, rng)
        output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1)
    else:
        predictions, _ = test_fn_deterministic(model, images, state)
        output = jax.nn.log_softmax(predictions, axis=-1)
    return (output.argmax(axis=-1) == labels.argmax(axis=-1)).mean(), predictions


@ eqx.filter_jit
def test_fn_memory(model: eqx.Module,
                   images: ndarray,
                   labels: ndarray,
                   rng,
                   state,
                   test_samples=None,):
    def compute_accuracies_predictions(images, labels, test_samples, model, state, rng):
        # First, we do a scan on the number of batches
        def scan_f(carry, data, model):
            image, label = data
            accuracy, predictions = compute_accuracy(
                model, image, label, state, test_samples, rng)
            return carry, (accuracy, predictions)

        _, (accuracies, predictions) = jax.lax.scan(
            f=ft.partial(scan_f, model=model),
            init=(),
            xs=(images, labels))
        return accuracies, predictions

    accuracies, predictions = compute_accuracies_predictions(
        images, labels, test_samples, model, state, rng)
    accuracies = expand_dims(accuracies.mean(), 0)
    predictions = predictions.reshape(
        predictions.shape[0] * predictions.shape[1], *predictions.shape[2:])
    return accuracies, predictions


@ eqx.filter_jit
def test_fn_permuted_mnist(model: eqx.Module,
                           images: ndarray,
                           labels: ndarray,
                           rng,
                           state,
                           max_parallel_permutation=1,
                           permutations=None,
                           test_samples=None,
                           norm_params=None):
    """ Complicated function here. Basically, we're defining batches over permutations of MNIST
    to increase the speed of testing of the different permutations which is the true bottleneck
    when willing to test on a large number of permutations. Then, we scan on the perm batches

    """

    def compute_accuracies_predictions(images, labels, test_samples, model, state, rng, norm_params):
        def scan_f(carry, data, norm_param, model):
            image, label = data
            if norm_param is not None:
                model = model.load_tree_norm(norm_param)
            accuracy, predictions = compute_accuracy(
                model, image, label, state, test_samples, rng)
            return carry, (accuracy, predictions)

        _, (accuracies, predictions) = jax.lax.scan(
            f=ft.partial(scan_f, norm_param=norm_params, model=model),
            init=(),
            xs=(images, labels))
        return accuracies, predictions

    if max_parallel_permutation < permutations.shape[0]:
        batched_permutations = permutations.reshape(
            max_parallel_permutation, permutations.shape[0] // max_parallel_permutation, *permutations.shape[1:])

        def reshape(x):
            return x.reshape(
                max_parallel_permutation, x.shape[0] // max_parallel_permutation, *x.shape[1:])
        norm_params = map(reshape, norm_params)
    else:
        batched_permutations = expand_dims(permutations, 0)
        norm_params = map(lambda x: expand_dims(x, 0), norm_params)

    def vmap_permutation(images, labels, test_samples, model, permutations, norm_params_batched):

        def scan_f_permutation(carry, data):
            permutation, norm_params_batch = data
            permuted_images = images.reshape(
                images.shape[0], images.shape[1], -1)[:, :, permutation].reshape(images.shape)
            accuracies, predictions = compute_accuracies_predictions(
                permuted_images, labels, test_samples, model, state, rng, norm_params_batch)
            return carry, (accuracies, predictions)

        _, (accuracies, predictions) = jax.lax.scan(
            f=scan_f_permutation,
            init=(),
            xs=(permutations, norm_params_batched))
        return accuracies, predictions
    accuracies, predictions = jax.vmap(vmap_permutation, in_axes=(
        None, None, None, None, 0, None if norm_params is None else 0))(images, labels, test_samples, model, batched_permutations, norm_params)
    accuracies = accuracies.reshape(
        accuracies.shape[0] * accuracies.shape[1], *accuracies.shape[2:]).mean(axis=1)
    predictions = predictions.reshape(
        predictions.shape[0] * predictions.shape[1], predictions.shape[2] * predictions.shape[3], *predictions.shape[4:])
    return accuracies, predictions
