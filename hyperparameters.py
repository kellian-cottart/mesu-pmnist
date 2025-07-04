import jax
import jax.numpy as jnp
from jax.numpy import expand_dims
import equinox as eqx
from models import *
from datetime import datetime
import os
import json
from optimizers import *
from utils import *
import argparse
import json
import optuna
from copy import deepcopy
from torch import manual_seed
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# first argument is name of config file
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file name",
                    type=str)
parser.add_argument(
    "-it", "--n_iterations", help="Number of iterations to run the hpo for", type=int, default=1)
parser.add_argument(
    "-gpu", "--gpu", help="GPU ID to use", type=int, default=0)
parser.add_argument(
    "-v", "--verbose", help="Verbose mode", action="store_true")
parser.add_argument(
    "-fits", "--fits_in_memory", help="Fits in memory", action="store_true")
args = parser.parse_args()

CONFIGURATION_LOADING_FOLDER = "hpo-configurations"
CONFIG_FILE = json.load(
    open(os.path.join(CONFIGURATION_LOADING_FOLDER, args.config+".json")))
N_ITERATIONS = args.n_iterations
CUDA_DEVICE = f"cuda:{args.gpu}"
VERBOSE = args.verbose
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S-")
MAIN_FOLDER = "study"
STUDY_NAME = f"{TIMESTAMP}{CONFIG_FILE['task']}-{CONFIG_FILE['network']}"
STUDY_FOLDER = os.path.join(MAIN_FOLDER, STUDY_NAME)
FITS_IN_MEMORY = args.fits_in_memory

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.makedirs(STUDY_FOLDER, exist_ok=True)


def compute_objective(accuracy, task, first_task_accuracies=None, weights=(0.5, 0.25, 0.25)):
    task_accuracy = accuracy[task].item()
    if len(accuracy) == 1:
        return task_accuracy*weights[0]
    mean_accuracy = jnp.mean(accuracy).item()
    if first_task_accuracies is None:
        return mean_accuracy*weights[1] + task_accuracy*weights[0]
    return task_accuracy*weights[2] + mean_accuracy*weights[1] + first_task_accuracies[0].item()*weights[0]


def assign_value(dict_type_fn, key_dict, key):
    if not isinstance(key_dict, dict):
        return key_dict
    type_fn = dict_type_fn[key_dict["type"]]
    if key_dict["type"] in ["float", "int"]:
        log = key_dict["step"] == 1
        key_dict = type_fn(
            key, low=key_dict["low"], high=key_dict["high"], step=key_dict["step"] if not log else None, log=log)
    else:
        key_dict = type_fn(
            key, low=key_dict["low"], high=key_dict["high"])
    return key_dict


def objective(trial):
    # Configurations is an array with N_ITERATIONS times the same config except with seed +=1
    # copy config_file without modyfing the original
    configuration = deepcopy(CONFIG_FILE)
    dict_type_fn = {
        "int": trial.suggest_int,
        "float": trial.suggest_float,
        "categorical": trial.suggest_categorical,
        "uniform": trial.suggest_uniform,
    }
    FOLDER = os.path.join(STUDY_FOLDER, f"trial-{trial.number}")
    os.makedirs(FOLDER, exist_ok=True)
    # retrieve the configuration parameters
    study_params = configuration["optimizer_params"]
    for keys in study_params.keys():
        # each key has a dictionnay with min and max indicating the range of the hyperparameter, and type indicating the type of the hyperparameter
        study_params[keys] = assign_value(
            dict_type_fn, study_params[keys], keys)
    configuration["optimizer_params"] = study_params
    # if ewc in configuration keys, we need to do the same
    # if has "ewc" in a key of configuration
    for keys in configuration.keys():
        if "ewc" in keys or keys == "si":
            ewc_params = configuration[keys]
            for ewc_key in ewc_params.keys():
                ewc_params[ewc_key] = assign_value(
                    dict_type_fn, ewc_params[ewc_key], ewc_key)
            configuration[keys] = ewc_params
    # Convert all fields to lowercase if string
    for k, v in configuration.items():
        if isinstance(v, str):
            configuration[k] = v.lower()
    ewc_parameters = None
    ewc_streaming_parameters = None
    ewc_online_parameters = None
    si_parameters = None
    if "stream-ewc" in configuration:
        ewc_streaming_parameters = deepcopy(configuration["stream-ewc"])
    elif "online-ewc" in configuration:
        ewc_online_parameters = deepcopy(configuration["online-ewc"])
    elif "ewc" in configuration:
        ewc_parameters = deepcopy(configuration["ewc"])
    # save config
    with open(FOLDER + "/config.json", "w") as f:
        json.dump(configuration, f, indent=4)

    # ----------------- LOADING THE DATASET -----------------
    # Initialize the random number generator
    manual_seed(configuration["seed"])
    np.random.seed(configuration["seed"])
    rng = jax.random.PRNGKey(configuration["seed"])
    # Load the dataset
    loader = GPULoading(device="cpu")
    task_params = configuration["task_params"] if "task_params" in configuration else {
    }
    max_permutations = configuration["max_parallel_permutation"] if "max_parallel_permutation" in configuration else 1
    train_samples = configuration["n_train_samples"] if "n_train_samples" in configuration else None
    test_samples = configuration["n_test_samples"] if "n_test_samples" in configuration else None
    train, test, shape, num_classes = loader.task_selection(
        configuration["task"], **task_params)
    val_key, rng = jax.random.split(rng)
    # ----------------- PERMUTATION IF PERMUTED MNIST -----------------
    is_permuted = configuration["task"] == "permutedmnist"
    permutations = None
    if is_permuted:
        perm_keys, rng = jax.random.split(rng, 2)
        perm_keys = jax.random.split(
            perm_keys, configuration["n_tasks"])
        permutations = jnp.array(
            [jax.random.permutation(key, jnp.array(shape).prod()) for key in perm_keys])
    # ----------------- CONFIGURING THE NETWORK/OPTIMIZER -----------------
    model_key, rng = jax.random.split(rng)
    # Configure the model
    model, model_state = configure_networks(configuration, model_key)
    print("Training on", configuration["task"])
    # Configure the optimizer
    optimizer, opt_state = configure_optimizer(
        configuration, eqx.filter(model, eqx.is_array))

    def initialize_ewc_parameters(model):
        old_param = eqx.filter(model, eqx.is_array)
        fisher = jax.tree.map(lambda x: jnp.zeros_like(x), old_param)
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
    # Synaptic metaplasticity imposes to store a normalization parameter at the end of each task
    # and to evaluate the model based on this normalization parameter
    # We pre-allocate the tree with the normalization parameters
    norm_params = None
    # ----------------- KEYGEN -----------------
    trkey, tekey, rng = jax.random.split(rng, 3)
    training_core_keys = jax.random.split(
        trkey, (configuration["n_tasks"], configuration["epochs"]))
    test_core_keys = jax.random.split(
        tekey, (configuration["n_tasks"], configuration["epochs"]))
    # ----------------- TRAINING/TESTING -----------------
    first_task_accuracies = None
    pbar = tqdm(range(configuration["n_tasks"]), desc="Tasks") if VERBOSE else range(
        configuration["n_tasks"])
    epoch_pbar = tqdm(range(configuration["epochs"]), desc="Epochs") if VERBOSE else range(
        configuration["epochs"])
    train = to_dataloader(
        train, configuration["train_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)
    test_dataloader = to_dataloader(
        test, configuration["test_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)
    for task_id, task in enumerate(pbar):
        if len(train) == configuration["n_tasks"]:
            task_train_dataloader = train[task_id]
        elif is_permuted:
            task_train_dataloader = reshape_perm(
                train[0], permutations[task_id])
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
            accuracies, _, _ = main_test_fn(
                test_dataset=test_dataloader,
                num_classes=num_classes,
                test_samples=test_samples,
                test_ck=test_ck,
                model=model,
                model_state=model_state,
                norm_params=None,
                is_permuted=is_permuted,
                max_permutations=max_permutations,
                permutations=permutations,
                fits_in_memory=FITS_IN_MEMORY
            )
            current_obj = compute_objective(
                accuracies, task_id, first_task_accuracies)
            current_step = task * configuration["epochs"] + epoch
            trial.report(current_obj, step=current_step)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()

            if VERBOSE:
                tqdm.write("=" * 20)
                for i, acc in enumerate(accuracies):
                    tqdm.write(f"{acc.item()*100:.2f}%", end="\t" if i % 10 !=
                               9 and i != len(accuracies) - 1 else "\n")
                # add loss to the bar
                tqdm.write(f"Loss: {jnp.mean(losses):.4f}")
                # write the objective
                tqdm.write(f"Objective: {current_obj:.4f}")
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
        if task_id == 0:
            first_task_accuracies = deepcopy(accuracies)
        # save accuracies as jax arrays
        with open(FOLDER + f"/accuracies-task={task_id}.npy", "wb") as f:
            jnp.save(f, accuracies)
    return compute_objective(accuracies, task_id, first_task_accuracies)


if __name__ == "__main__":
    print("Starting study with name: ", STUDY_NAME)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_FOLDER}/{STUDY_NAME}.sqlite3",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=10),
        pruner=None)
    study.optimize(objective, n_trials=N_ITERATIONS)
    # save the best study number
    print("Best trial: ", study.best_trial.number)
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    with open(os.path.join(STUDY_FOLDER, "best_study.txt"), "w") as f:
        f.write(str(study.best_trial.number) +
                ": " + str(study.best_trial.value))
        json.dump(study.best_params, f, indent=4)
