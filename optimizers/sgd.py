import jax
import optax
from jax.tree import map

def sgd(lr: float = 0.001,):
    """
    SGD

    Args:
        lr: Learning rate for the optimizer.
    """

    def init(params):
        return {
            'step': 0,
        }

    def update(gradients, state, params=None):
        state['step'] += 1
        # Update the parameters using sgd
        return map(lambda g: -lr * g, gradients), state

    return optax.GradientTransformation(init, update)
