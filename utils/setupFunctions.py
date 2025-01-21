import jax
from models import *
from optimizers import *
import optax
from equinox.nn import make_with_state


def configure_networks(configuration, rng):
    # make a dictionary of maps
    select_network = {
        "bayesianmlp": BaseBayesianMLP,
        "bayesianmlplayernorm": BayesianMLPLayerNorm,
        "mlp": BaseMLP,
        "mlpbatchnorm": MLPBatchNorm,
        "mlplayernorm": MLPLayerNorm,
    }
    if not "network_params" in configuration:
        raise ValueError("Network parameters 'network_params' not found")
    if not "activation_fn" in configuration["network_params"]:
        configuration["network_params"]["activation_fn"] = "relu"

    select_activation = {
        "relu": jax.nn.relu,
    }
    # Certain activation functions don't take any parameters, so we try defaulting to the function alone
    try:
        configuration["network_params"]["activation_fn"] = select_activation[configuration["network_params"]["activation_fn"]](
            **configuration["network_params"]["activation_params"]
        )
    except (TypeError, KeyError) as e:
        try:
            configuration["network_params"]["activation_fn"] = select_activation[configuration["network_params"]["activation_fn"]]()
        except TypeError as e:
            configuration["network_params"]["activation_fn"] = select_activation[configuration["network_params"]["activation_fn"]]

    # Instantiate the model and the initial state
    try:
        key, rng = jax.random.split(rng, 2)
        # make_with_state separates the model and the initial state for batch norm tracking stats
        model, model_state = make_with_state(select_network[configuration["network"]])(
            key=key, **configuration["network_params"])
    except KeyError as e:
        raise KeyError("Error with provided keys: ", e)

    return model, model_state


def configure_optimizer(configuration, model):
    select_optimizer = {
        "sgd": sgd,
        "adam": optax.adam,
        "mesu": mesu,
        "bgd": bgd,
    }
    if not "optimizer_params" in configuration:
        raise ValueError("Optimizer parameters not found")
    try:
        optimizer = select_optimizer[configuration["optimizer"]](
            **configuration["optimizer_params"]
        )
    except KeyError as e:
        raise KeyError("Error with provided keys: ", e)
    opt_state = optimizer.init(model)
    return optimizer, opt_state
