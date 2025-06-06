from jax.numpy import shape, dot, ones
from typing import Literal, Union
from jaxtyping import PRNGKeyArray, Array
from math import sqrt
from equinox import Module, field
from equinox import _misc
from .presynapticParameter import PresynapticParameter
from jax.random import split, normal, uniform


class PresynapticLinear(Module, strict=True):
    """Performs a linear transformation."""

    weight: dict[str, Array]
    bias: dict[str, Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """ Initialises the Bayesian Linear Layer

        Args:
            in_features: The input size. The input to the layer should be a vector of
                shape `(in_features,)`
            out_features: The output size. The output from the layer will be a vector
                of shape `(out_features,)`.
            use_bias: Whether to add on a bias as well.
            dtype: The dtype to use for the weight and the bias in this layer.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
                on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for GaussianParameter
                initialisation. (Keyword only argument.)
        """
        dtype = _misc.default_floating_dtype() if dtype is None else dtype
        wkey, bkey = split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / sqrt(in_features_)
        wshape = (out_features_, in_features_)
        bshape = (out_features_,)
        self.weight = PresynapticParameter(
            presynaptic=uniform(wkey, wshape, minval=-lim, maxval=lim),
            strength=uniform(wkey, wshape, minval=-lim, maxval=lim),
        )
        self.bias = PresynapticParameter(
            presynaptic=uniform(bkey, bshape, minval=-lim,
                       maxval=lim) if use_bias else None,
            strength=uniform(bkey, bshape, minval=-lim,
                       maxval=lim) if use_bias else None,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array) -> Array:
        """ Call function for bayesian linear layer
        `samples` forward passes using weights reparametrization w = mu + sigma * epsilon, epsilon ~ N(0, 1)

        Args:
            x: input tensor
            samples: number of samples
            rng: random key
        """
        weights = self.weight.presynaptic * self.weight.strength 
        output = dot(weights, x)
        if self.use_bias:
            biases = self.bias.presynaptic * self.bias.strength
            output = output + biases
        return output
