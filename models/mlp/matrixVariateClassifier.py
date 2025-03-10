from customLayers import MatrixVariateLinear
from equinox import Module
from jax.random import split
from jax.nn import relu
from jax.numpy import ravel
from jax import vmap
from equinox.nn import LayerNorm, BatchNorm

import jax


class BaseMatrixVariateMLP(Module):
    layers: list[MatrixVariateLinear]

    def __init__(self, key, layers=[784, 50, 10], use_bias=False, activation_fn=None, norm_fn=None, norm_params=None, **kwargs):
        keys = split(key, len(layers) - 1)
        # Start with `ravel`
        self.layers = []
        # Add `BayesianLinear` and `relu` alternately except after the last linear layer
        for i in range(len(layers) - 1):
            # Add `BayesianLinear` layer
            self.layers.append(MatrixVariateLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=use_bias,
                key=keys[i]
            ))
            if norm_fn == LayerNorm:
                norm_params["shape"] = (layers[i + 1],)
                self.layers.append(norm_fn(**norm_params))
            elif norm_fn == BatchNorm:
                norm_params["input_size"] = layers[i + 1]
                self.layers.append(norm_fn(**norm_params))
            # Add `relu` if not the last layer
            if i < len(layers) - 2:
                self.layers.append(activation_fn)

    def __call__(self, x, state, samples, key, *, backwards=False):
        x = ravel(x)
        if backwards:
            for layer in self.layers:
                x = layer(x)
        else:
            s_key = split(key, samples)
            def forward(x, layers, s_l_key):
                for layer in layers:
                    if isinstance(layer, MatrixVariateLinear):
                        l_key, s_l_key = split(s_l_key, 2)
                        x = layer.sample(x, key=l_key)
                    else:
                        x = layer(x)
                return x
            x = vmap(forward, in_axes=(None, None, 0))(x, self.layers, s_key)

        return x, state

