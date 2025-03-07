import jax
import jax.numpy as jnp
from jax.numpy import expand_dims
import equinox as eqx
from models import *
from datetime import datetime
import os
import json
from shutil import rmtree
from optimizers import *
import traceback
from utils import *
import argparse
import json
from copy import deepcopy
import numpy as np
from torch import manual_seed, prod, tensor
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
# argparse allows to load a configuration from a file
CONFIGURATION_LOADING_FOLDER = "configurations"
# first argument is name of config file
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file name",
                    type=str)
parser.add_argument(
    "-it", "--n_iterations", help="Number of iterations to run the config file for", type=int, default=1)
parser.add_argument(
    "-v", "--verbose", help="Whether to display the pbar or not", action="store_true")
parser.add_argument(
    "-ood", "--ood", help="Which dataset to compute the ood on (fashion, pmnist, None)", type=str, default=None)
parser.add_argument(
    "-gpu", "--gpu", help="GPU ID to use", type=int, default=0)
parser.add_argument(
    "-wh", "--weight_histogram", help="Whether to save weight histograms", action="store_true")
parser.add_argument(
    "-fits", "--fits_in_memory", help="Whether the dataset fits in memory or not", action="store_true")
args = parser.parse_args()
CONFIG_FILE = json.load(
    open(os.path.join(CONFIGURATION_LOADING_FOLDER, args.config+".json")))
for k, v in CONFIG_FILE.items():
    if isinstance(v, str):
        CONFIG_FILE[k] = v.lower()
N_ITERATIONS = args.n_iterations
OOD = args.ood
VERBOSE = args.verbose
WEIGHT_HIST = args.weight_histogram
FITS_IN_MEMORY = args.fits_in_memory
# set the device
jax.config.update("jax_platform_name", "gpu")

if __name__ == "__main__":
    # Create a timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S-")
    MAIN_FOLDER = "results"
    os.makedirs(MAIN_FOLDER, exist_ok=True)
    for k in range(N_ITERATIONS):
        configuration = deepcopy(CONFIG_FILE)
        configuration["seed"] += k
        print(f"========== CONFIGURATION {k} ==========")
        print(json.dumps(configuration, indent=4))
        print("========================================")
        FOLDER = f"{TIMESTAMP}{configuration['task']}-t={configuration['n_tasks']}-e={configuration['epochs']}-opt={configuration['optimizer']}"
        ewc_parameters = None
        ewc_streaming_parameters = None
        ewc_online_parameters = None
        si_parameters = None
        if "stream-ewc" in configuration:
            ewc_streaming_parameters = deepcopy(configuration["stream-ewc"])
            FOLDER += f"-stream-ewc={configuration['stream-ewc']['importance']}"
        if "online-ewc" in configuration:
            ewc_online_parameters = deepcopy(configuration["online-ewc"])
            FOLDER += f"-online-ewc={configuration['online-ewc']['importance']}"
        if "ewc" in configuration:
            ewc_parameters = deepcopy(configuration["ewc"])
            FOLDER += f"-ewc={configuration['ewc']['importance']}"
        if "si" in configuration:
            si_parameters = deepcopy(configuration["si"])
            FOLDER += f"-si={configuration['si']['coefficient']}"
        if "use_bias" in configuration["network_params"]:
            FOLDER += "-bias"
        SAVE_PATH = os.path.join(MAIN_FOLDER, FOLDER)
        CONFIGURATION_PATH = os.path.join(SAVE_PATH, f"config{k}")
        DATA_PATH = os.path.join(CONFIGURATION_PATH, "accuracy")
        WEIGHTS_PATH = os.path.join(CONFIGURATION_PATH, "weights")
        UNCERTAINTY_PATH = os.path.join(CONFIGURATION_PATH, "uncertainty")
        for path in [SAVE_PATH, CONFIGURATION_PATH, DATA_PATH, WEIGHTS_PATH, UNCERTAINTY_PATH]:
            os.makedirs(path, exist_ok=True)
        # save config
        with open(CONFIGURATION_PATH + "/config.json", "w") as f:
            json.dump(configuration, f, indent=4)
        try:
            # Initialize the random number generator
            manual_seed(configuration["seed"])
            np.random.seed(configuration["seed"])
            rng = jax.random.PRNGKey(configuration["seed"])
            # Load the dataset
            loader = GPULoading(device="cpu")
            dataset_normalisation = configuration[
                "dataset_normalisation"] if "dataset_normalisation" in configuration else None
            max_permutations = configuration["max_parallel_permutation"] if "max_parallel_permutation" in configuration else 1
            train_samples = configuration["n_train_samples"] if "n_train_samples" in configuration else None
            test_samples = configuration["n_test_samples"] if "n_test_samples" in configuration else None
            train, test, shape, num_classes = loader.task_selection(
                configuration["task"], dataset_normalisation=dataset_normalisation)
            is_permuted = configuration["task"] == "permutedmnist"
            norm_params = None
            # Permutations
            permutations = None
            if is_permuted:
                perm_keys, rng = jax.random.split(rng, 2)
                perm_keys = jax.random.split(
                    perm_keys, configuration["n_tasks"])
                permutations = jnp.array(
                    [jax.random.permutation(key, jnp.array(shape).prod()) for key in perm_keys])

            model_key, rng = jax.random.split(rng)
            # Configure the model
            model, model_state = configure_networks(configuration, model_key)
            print("Training on", configuration["task"])
            # Configure the optimizer
            optimizer, opt_state = configure_optimizer(
                configuration, eqx.filter(model, eqx.is_array))

            def initialize_ewc_parameters(model):
                old_param = eqx.filter(model, eqx.is_array)
                fisher = map(lambda x: jnp.zeros_like(x), old_param)
                return old_param, fisher

            if ewc_parameters is not None:
                old_param, fisher = initialize_ewc_parameters(model)
                old_param = map(lambda x: expand_dims(x, 0).repeat(
                    configuration["n_tasks"], axis=0), old_param)
                fisher = map(lambda x: expand_dims(x, 0).repeat(
                    configuration["n_tasks"], axis=0), fisher)
                ewc_parameters["old_param"], ewc_parameters["fisher"] = old_param, fisher
            if ewc_online_parameters is not None:
                ewc_online_parameters["old_param"], ewc_online_parameters["fisher"] = initialize_ewc_parameters(
                    model)
            if ewc_streaming_parameters is not None:
                ewc_streaming_parameters["old_param"], ewc_streaming_parameters["fisher"] = initialize_ewc_parameters(
                    model)

            def initialize_si_parameters(model):
                old_param = eqx.filter(model, eqx.is_array)
                omega = map(lambda x: jnp.zeros_like(x), old_param)
                w_k = map(lambda x: jnp.zeros_like(x), old_param)
                return old_param, omega, w_k

            if si_parameters is not None:
                si_parameters["old_param"], si_parameters["omega"], si_parameters["w_k"] = initialize_si_parameters(
                    model)

            # GENERATING A HUGE ARRAY OF KEYS, ASSURING THAT THE KEYS ARE UNIQUE
            trkey, tekey, rng = jax.random.split(rng, 3)
            training_core_keys = jax.random.split(
                trkey, (configuration["n_tasks"], configuration["epochs"]))
            test_core_keys = jax.random.split(
                tekey, (configuration["n_tasks"], configuration["epochs"]))
            pbar = tqdm(range(configuration["n_tasks"]), desc="Tasks") if VERBOSE else range(
                configuration["n_tasks"])
            epoch_pbar = tqdm(range(configuration["epochs"]), desc="Epochs") if VERBOSE else range(
                configuration["epochs"])
            if OOD is not None:
                if "fashion" in OOD:
                    _, test_ood, shape_ood, num_classes_ood = loader.task_selection(
                        "fashion")
                elif "pmnist" in OOD:
                    _, test_ood, shape_ood, num_classes_ood = loader.task_selection(
                        "mnist")
                    ood_permutation = randperm(prod(tensor(shape_ood)))
                    images, labels = test_ood[0][:]
                    images = images.reshape(
                        images.shape[0], -1)[:, ood_permutation].reshape(images.shape)
                    test_ood[0] = TensorDataset(images, labels)
                ood_dataloader = to_dataloader(
                    test_ood, configuration["test_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)
                ookey, rng = jax.random.split(rng)
                ood_core_keys = jax.random.split(
                    ookey, (configuration["n_tasks"], configuration["epochs"]))
                # Prepare the dataloader
            train = to_dataloader(
                train, configuration["train_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)
            test_dataloader = to_dataloader(
                test, configuration["test_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)
            for task_id, task in enumerate(pbar):
                if is_permuted:
                    task_train_dataloader = reshape_perm(
                        train[0], permutations[task_id])
                elif len(train) == configuration["n_tasks"]:
                    task_train_dataloader = train[task_id]
                else:
                    raise ValueError("Length of train and n_tasks do not match: ", len(
                        train), " != ", configuration["n_tasks"])
                for epoch in epoch_pbar:
                    if VERBOSE:
                        pbar.set_description(
                            f"Task {task+1}/{configuration['n_tasks']} - Epoch {epoch+1}/{configuration['epochs']}")
                    train_ck = training_core_keys[task_id, epoch]
                    test_ck = test_core_keys[task_id, epoch]
                    model, opt_state, losses, ewc_streaming_parameters, model_state, si_parameters = train_fn(
                        model=model,
                        dataset=task_train_dataloader,
                        num_classes=num_classes,
                        opt_state=opt_state,
                        optimizer=optimizer,
                        train_ck=train_ck,
                        train_samples=train_samples,
                        ewc_online_parameters=ewc_online_parameters,
                        ewc_streaming_parameters=ewc_streaming_parameters,
                        ewc_parameters=ewc_parameters,
                        si_parameters=si_parameters,
                        init_state=model_state,
                    )
                    accuracies, predictions = main_test_fn(
                        test_dataset=test_dataloader,
                        num_classes=num_classes,
                        test_samples=test_samples,
                        test_ck=test_ck,
                        model=model,
                        model_state=model_state,
                        norm_params=norm_params,
                        is_permuted=is_permuted,
                        max_permutations=max_permutations,
                        permutations=permutations,
                        fits_in_memory=FITS_IN_MEMORY
                    )
                    if VERBOSE:
                        for i, acc in enumerate(accuracies):
                            tqdm.write(f"{acc.item()*100:.2f}%", end="\t" if i % 10 !=
                                       9 and i != len(accuracies) - 1 else "\n")
                        # add loss to the bar
                        tqdm.write(f"Loss: {jnp.mean(losses):.4f}")
                    if WEIGHT_HIST:
                        # save  weights
                        filter_weights = eqx.filter(model, eqx.is_array)
                        leaves = jax.tree.leaves(filter_weights)
                        for i, leaf in enumerate(leaves):
                            with open(os.path.join(WEIGHTS_PATH, f"layer={i}-task={task_id}-epoch={epoch}.npy"), "wb") as f:
                                jnp.save(f, leaf)
                    aleatoric_u, epistemic_u = compute_uncertainty(
                        predictions[task_id])
                    for metric_name, metric_data in {"aleatoric": aleatoric_u, "epistemic": epistemic_u}.items():
                        np.save(os.path.join(
                                UNCERTAINTY_PATH, f"{metric_name}-task={task_id}-epoch={epoch}.npy"), metric_data)
                    if OOD is not None:
                        # Compute uncertainty
                        ood_k = ood_core_keys[task_id, epoch]
                        ood_accuracies, ood_predictions = main_test_fn(
                            test_dataset=ood_dataloader,
                            num_classes=num_classes,
                            test_samples=test_samples,
                            test_ck=ood_k,
                            model=model,
                            model_state=model_state,
                            norm_params=norm_params,
                            is_permuted=False,
                            fits_in_memory=FITS_IN_MEMORY
                        )
                        ood_aleatoric_u, ood_epistemic_u = compute_uncertainty(
                            ood_predictions[0])
                        epistemic_roc_auc = compute_roc_auc(
                            epistemic_u, ood_epistemic_u)
                        aleatoric_roc_auc = compute_roc_auc(
                            aleatoric_u, ood_aleatoric_u)
                        uncertainty_metrics = {
                            "ood-aleatoric": ood_aleatoric_u,
                            "ood-epistemic": ood_epistemic_u,
                            "roc-auc-aleatoric": aleatoric_roc_auc,
                            "roc-auc-epistemic": epistemic_roc_auc
                        }
                        for metric_name, metric_data in uncertainty_metrics.items():
                            np.save(os.path.join(
                                UNCERTAINTY_PATH, f"{metric_name}-task={task}-epoch={epoch}.npy"), metric_data)
                    with open(os.path.join(DATA_PATH, f"task={task}-epoch={epoch}.npy"), "wb") as f:
                        jnp.save(f, accuracies)
                # ewc requires saving at the end of the task the current model parameters
                if ewc_parameters is not None or ewc_online_parameters is not None:
                    fisher = compute_fisher(
                        model=model,
                        dataset=task_train_dataloader,
                    )
                    if ewc_online_parameters is not None:
                        ewc_online_parameters["old_param"] = eqx.filter(
                            model, eqx.is_array)
                        ewc_online_parameters["fisher"] = map(
                            lambda old, new: ewc_online_parameters["downweighting"] * old + new, ewc_online_parameters["fisher"], fisher)
                    elif ewc_parameters is not None:
                        ewc_parameters["old_param"] = map(lambda old, new: old.at[task_id].set(new),
                                                          ewc_parameters["old_param"], eqx.filter(model, eqx.is_array))
                        ewc_parameters["fisher"] = map(lambda old, new: old.at[task_id].set(new),
                                                       ewc_parameters["fisher"], fisher)
                if si_parameters is not None:
                    epsilon = si_parameters["damping_factor"]
                    difference = map(lambda old, new: (
                        new - old)**2, si_parameters["old_param"], eqx.filter(model, eqx.is_array))
                    si_parameters["omega"] = map(lambda omega, diff, w: omega + relu(w/(diff + epsilon)),
                                                 si_parameters["omega"],
                                                 difference,
                                                 si_parameters["w_k"])
                    si_parameters["w_k"] = map(
                        lambda x: jnp.zeros_like(x), si_parameters["w_k"])
                    si_parameters["old_param"] = eqx.filter(
                        model, eqx.is_array)

        except (KeyboardInterrupt, SystemExit, Exception):
            print(traceback.format_exc())
            rmtree(SAVE_PATH)
