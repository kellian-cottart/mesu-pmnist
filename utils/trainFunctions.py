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


@eqx.filter_jit
def loss_fn(model, images, labels, samples=None, rng=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, init_state=None):
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
    elif any(param is not None for param in [ewc_online_parameters, ewc_streaming_parameters, ewc_parameters]):
        loss_fn_to_use = ewc_loss_fn
        ewc_params = ewc_online_parameters or ewc_streaming_parameters or ewc_parameters
        loss_args = (model, images, labels, ewc_params, init_state)
    else:
        loss_fn_to_use = deterministic_loss_fn
        loss_args = (model, images, labels, init_state)
    return loss_fn_to_use(*loss_args)


@ eqx.filter_value_and_grad(has_aux=True)
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


def bayesbinn_loss_fn(model, images, labels, samples, rng, init_state=None):
    """ Loss function for Bayesian models. """
    # Split the random key for each sample
    samples_rng = jax.random.split(rng, samples)

    def closure(model, samples_rng, init_state):
        # Create a structure copy of the model
        struct_copy = structure(model)
        # Split the random keys for each leaf in the model
        rkeys = split(samples_rng, struct_copy.num_leaves)
        # Partition the model into dynamic and static parts
        dynamic, static = eqx.partition(model, eqx.is_array)

        def make_noise(x, l_key):
            # Generate uniform noise and apply logit transformation
            uniform_noise = uniform(
                l_key, x.shape, minval=1e-10, maxval=1-1e-10)
            logits = log(uniform_noise) - log(1 - uniform_noise)
            return 0.5 * logits

        def reparam_model_dynamic(x, noise, temperature=1):
            # Apply reparameterization trick with tanh
            return tanh((x + noise) / temperature)

        # Generate noise for the dynamic part of the model
        noise_tree = map(ft.partial(make_noise), dynamic,
                         unflatten(struct_copy, rkeys))
        # Reparameterize the dynamic part of the model
        reparam_model_dynamic = map(ft.partial(
            reparam_model_dynamic, temperature=model.temperature), dynamic, noise_tree)
        # Combine the reparameterized dynamic part with the static part
        reparam_model = eqx.combine(reparam_model_dynamic, static)

        @ eqx.filter_value_and_grad(has_aux=True)
        def loss_fn(model, images, labels, init_state):
            # Compute predictions using the reparameterized model
            predictions, state = jax.vmap(ft.partial(model, backwards=True), axis_name="batch", in_axes=(
                0, None, None, None), out_axes=(0, None))(images, init_state, samples, samples_rng)
            # Compute the log softmax of the predictions
            output = jax.nn.log_softmax(predictions, axis=-1) * labels
            # Compute the loss
            loss = -jnp.sum(output, axis=-1).sum()
            return loss, state
        # Compute the loss and gradients
        (losses, state), grads = loss_fn(
            reparam_model, images, labels, init_state)
        return losses, grads, noise_tree, state

    # Vectorize the closure function over the samples
    losses, grads, noise_tree, state = jax.vmap(
        closure, in_axes=(None, 0, None), out_axes=(0, 0, 0, None))(model, samples_rng, init_state)

    def grad_on_mu(grad, x, noise, temperature=1):
        # Compute the gradient with respect to the mean
        mu = tanh(x)
        gumbel = tanh((x + noise) / temperature)
        derivative_gumbel = 1 - gumbel * gumbel + 1e-7
        derivative_mu = 1 - mu * mu + 1e-7
        dwdmu = derivative_gumbel / (temperature * derivative_mu)
        output = dwdmu * grad
        return output.mean(axis=0)
    # Adjust the gradients
    grads = jax.tree.map(ft.partial(grad_on_mu, temperature=model.temperature), grads, eqx.filter(
        model, eqx.is_array), eqx.filter(noise_tree, eqx.is_array))
    return (jnp.mean(losses), state), grads


@ eqx.filter_value_and_grad(has_aux=True)
def bayesian_loss_fn(model, images, labels, samples, rng, init_state=None):
    """ Loss function for Bayesian models. """
    # Same rng for all images in the batch, but different for each sample
    predictions, state = jax.vmap(ft.partial(model, backwards=True),
                                  in_axes=(0, None, None, None), out_axes=(0, None))(images, init_state, samples, rng, )
    output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1) * labels
    loss = -jnp.sum(output, axis=-1).sum()
    return loss, state


@ eqx.filter_value_and_grad(has_aux=True)
def deterministic_loss_fn(model, images, labels, init_state=None):
    """ Loss function for deterministic models. """
    predictions, state = jax.vmap(model, axis_name="batch",
                                  in_axes=(0, None), out_axes=(0, None))(images, init_state)
    output = jax.nn.log_softmax(predictions, axis=-1) * labels
    # mean reduction only on deterministic models
    loss = -jnp.sum(output, axis=-1).mean()
    return loss, state


def train_fn(model, dataset, num_classes, opt_state, optimizer, train_ck, train_samples=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, init_state=None):
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
        "init_state": init_state
    }
    if isinstance(dataset, DataLoader):
        return dataloader_train_fn(**args)
    else:
        return all_gpu_train_fn(**args)


def dataloader_train_fn(model, dataset, num_classes, opt_state, optimizer, train_ck, train_samples=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, init_state=None):
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
            init_state=init_state)
        # Update the model using the optimizer
        updates, opt_state = optimizer.update(
            grads, opt_state, dynamic_state)
        dynamic_state = optax.apply_updates(dynamic_state, updates)
        model = eqx.combine(dynamic_state, static_state)
        if ewc_streaming_parameters is not None:
            ewc_streaming_parameters["old_param"] = eqx.filter(
                model, eqx.is_array)
            ewc_streaming_parameters = update_ewc_streaming_parameters(
                model, images, labels, train_samples, ewc_streaming_parameters)
        losses.append(loss)
    losses = jnp.array(losses)
    return model, opt_state, losses, ewc_streaming_parameters, init_state


@ eqx.filter_jit
def all_gpu_train_fn(model, dataset, num_classes, opt_state, optimizer, train_ck, train_samples=None, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None, init_state=None):
    """ Train the model on the task.
    Splits the model into dynamic and static parts to allow for eqx.filter_jit compilation.
    """
    task_train_images = dataset[0]
    task_train_labels = dataset[1]
    dynamic_init_state, static_state = eqx.partition(model, eqx.is_array)

    def scan_fn(carry, data):
        dynamic_state, opt_state, ewc_streaming_parameters, init_state = carry
        images, labels, key = data
        # Train the model
        dynamic_state, opt_state, loss, ewc_streaming_parameters, init_state = batch_fn(
            dynamic_state, opt_state, key, optimizer, images, labels, init_state, train_samples, static_state, ewc_online_parameters, ewc_streaming_parameters, ewc_parameters)
        return (dynamic_state, opt_state, ewc_streaming_parameters, init_state), loss

    # Split the random key for each batch of data
    train_ck = jax.random.split(train_ck, task_train_images.shape[0])
    # Use jax.lax.scan to iterate over the batches
    (dynamic_init_state, opt_state, ewc_streaming_parameters, init_state), losses = jax.lax.scan(
        f=scan_fn,
        init=(dynamic_init_state, opt_state,
              ewc_streaming_parameters, init_state),
        xs=(task_train_images, task_train_labels, train_ck)
    )
    # Combine the dynamic and static parts of the model to recover the activation functions
    model = eqx.combine(dynamic_init_state, static_state)
    return model, opt_state, losses, ewc_streaming_parameters, init_state

def batch_fn(dynamic_state, opt_state, keys, optimizer, images, labels, state, samples, static_state, ewc_online_parameters=None, ewc_streaming_parameters=None, ewc_parameters=None):
    # Combine dynamic and static parts of the model
    model = eqx.combine(dynamic_state, static_state)
    (loss, state), grads = loss_fn(
        model, images, labels, samples, keys, ewc_online_parameters, ewc_streaming_parameters, ewc_parameters, state)
    # Update the model using the optimizer
    dynamic_state = eqx.partition(model, eqx.is_array)[0]
    updates, opt_state = optimizer.update(grads, opt_state, dynamic_state)
    dynamic_state = optax.apply_updates(dynamic_state, updates)
    if ewc_streaming_parameters is not None:
        ewc_streaming_parameters["old_param"] = eqx.filter(model, eqx.is_array)
        ewc_streaming_parameters = update_ewc_streaming_parameters(
            model, images, labels, samples, ewc_streaming_parameters)
    return dynamic_state, opt_state, loss, ewc_streaming_parameters, state

def update_ewc_streaming_parameters(model, images, labels, samples, ewc_streaming_parameters):
    # Compute the new gradients for the fisher information matrix
    _, fisher_grads = loss_fn(model, images, labels, samples)
    fisher = jax.tree.map(lambda x: x ** 2, fisher_grads)
    ewc_streaming_parameters["fisher"] = jax.tree.map(
        lambda old, new: ewc_streaming_parameters["downweighting"] * old + new, ewc_streaming_parameters["fisher"], fisher)
    return ewc_streaming_parameters


@ eqx.filter_jit
def compute_fisher(model, dataset):
    def fisher_batch_fn(images, labels):
        # Compute the loss and gradients
        _, grads = loss_fn(model, images, labels)
        # Compute the Fisher information matrix by squaring the gradients
        return jax.tree.map(lambda x: x ** 2, grads)

    def scan_fn(carry, data):
        images, labels = data
        # Train the model
        fisher = fisher_batch_fn(images, labels)
        # Return Princess Leia
        return carry, fisher
    # Split the dataset into images and labels
    task_train_images, task_train_labels = dataset
    # Compute the new fisher information matrix
    _, fisher = jax.lax.scan(
        f=scan_fn,
        init=(),
        xs=(task_train_images, task_train_labels)
    )
    # Compute expectation of the empirical Fisher information matrix
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), fisher)
