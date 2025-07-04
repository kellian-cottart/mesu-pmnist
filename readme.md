# Bayesian continual learning and forgetting in neural networks - MNIST/PermutedMNIST Experiments

## Description

This is the repository holding the code used to generate the results of the paper. Here is given the environment and

## Environment installation

To install the environment needed to run the code, please use Conda/Mamba/Micromamba. Make sure you are running either of the commands in the main directory.

```bash
conda env create -f environment.yml
conda activate bayesian
```

## Main figures

The main file is `main.py`. It is used to run the experiments according to the configuration file. The configuration file is a JSON file that contains the parameters for the experiment. Namely, the network architecture, the hyperparameters, the dataset, the optimizer. Here are the different arguments that can be passed to the main file:

* `-c` or `--config_file`: The path to the configuration file;
* `-it` or `--n_iterations`: The number of iterations to run the experiment for;
* `-v` or `--verbose`: Whether to display the progress bar or not;
* `-ood` or `--ood`: The dataset to compute the OOD on. Available datasets are `fashion`, `pmnist`, `None`;
* `-gpu` or `--gpu`: The GPU ID to use;
* `-wh` or `--weight_histogram`: Whether to save the weight histograms at each epochs;
* `-fits` or `--fits_in_memory`: Whether the dataset fits in GPU memory or not. Else, the dataset is loaded in RAM.

### Figure PMNIST

These are the commands for Fig. 3. Each command runs for about 50 minutes for non-Bayesian neural networks and 2h30 for Bayesian networks on a RTX 3090.

```bash
python main.py -it 5 -ood fashion -c pmnist/mlp-sgd -fits
python main.py -it 5 -ood fashion -c pmnist/mlp-online-ewc-sgd -fits
python main.py -it 5 -ood fashion -c pmnist/mlp-stream-ewc-sgd -fits
python main.py -it 5 -ood fashion -c pmnist/mlp-bayesian-mesu -fits
python main.py -it 5 -ood fashion -c pmnist/mlp-bayesian-foovbdiagonal -fits
python main.py -it 5 -ood fashion -c pmnist/mlp-si-sgd -fits
```

### Figure MNIST OOD

These are the commands for Fig. 4. Each command runs for about 50 minutes for non-Bayesian neural networks and 2h30 for Bayesian networks on a RTX 3090.

```bash
python main.py -it 5 -ood pmnist -wh -c mnist-ood/mlp-sgd -fits
python main.py -it 5 -ood pmnist -wh -c mnist-ood/mlp-stream-ewc-sgd -fits
python main.py -it 5 -ood pmnist -wh -c mnist-ood/mlp-bayesian-foovbdiagonal -fits
python main.py -it 5 -ood pmnist -wh -c mnist-ood/mlp-bayesian-mesu -fits
python main.py -it 5 -ood pmnist -wh -c mnist-ood/mlp-bayesian-mesu-high-N -fits
```

## Supplementary figures

### Figure PMNIST - Study of hyperparameter N

```bash
python main.py -it 5 -ood fashion -wh -c supp-n-study/mlp-bayesian-mesu-N=180000 -fits
python main.py -it 5 -ood fashion -wh -c supp-n-study/mlp-bayesian-mesu-N=300000 -fits
python main.py -it 5 -ood fashion -wh -c supp-n-study/mlp-bayesian-mesu-N=600000 -fits
python main.py -it 5 -ood fashion -wh -c supp-n-study/mlp-bayesian-mesu-N=1200000 -fits
python main.py -it 5 -ood fashion -wh -c supp-n-study/mlp-bayesian-mesu-N=2400000 -fits
python main.py -it 5 -ood fashion -wh -c supp-n-study/mlp-bayesian-mesu-N=1e15 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=256-N=180000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=256-N=300000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=256-N=600000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=256-N=1200000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=256-N=2400000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=256-N=1e15 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=512-N=180000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=512-N=300000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=512-N=600000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=512-N=120000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=512-N=2400000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=512-N=1e15 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=1024-N=180000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=1024-N=300000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=1024-N=600000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=1024-N=1200000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=1024-N=2400000 -fits
python main.py -it 5 -ood fashion -wh -c supp-size-study/mlp-bayesian-mesu-S=1024-N=1e15 -fits
```

### Figure PMNIST - Presynaptic Consolidation

```bash
python main.py -it 5 -c supp-presynaptic/mlp-bayesian-mesu -fits
```

### Figure PMNIST - UCB

```bash
python main.py -it 5 -c supp-ucb/mlp-bayesian-mesu-small -fits
```

## Notebooks generating figures

### Main figures

Two files are available in the repository:

- [Figure PMNIST](./figure-pmnist.ipynb)
- [Figure MNIST OOD](./figure-mnist-ood.ipynb)

These two files are linked with two folders, respectively `RESULTS-PMNIST` and `RESULTS-MNIST-OOD`. To retrieve the figures, these folders must contain the results of the experiments previously generated with the figure commands.

The resulting figures will be saved in a folder named `output-figures` in the main directory.

### Supplementary figures

The supplementary figures are generated in the same way as the main figures. The notebooks are:
- [Figure PMNIST - Study of hyperparameter N](./figure-pmnist-N-study.ipynb)
- [Figure PMNIST - Presynaptic Consolidation](./figure-pmnist-presynaptic.ipynb)
- [Figure PMNIST - UCB](./figure-pmnist-UCB.ipynb)

The results of the experiments must be saved in the `RESULTS-N-STUDY` folder for the first notebook, and in the `RESULTS-PRESYNAPTIC` and `RESULTS-UCB` folders for the other two notebooks.

## Authors

- [Djohan BONNET](https://scholar.google.com/citations?user=1cSwOPIAAAAJ&hl=en)
- [Kellian COTTART](https://scholar.google.com/citations?hl=en&user=Akg-AH4AAAAJ)

## Citation

Please reference this work as

@misc{bonnet2025bayesiancontinuallearningforgetting,
      title={Bayesian continual learning and forgetting in neural networks}, 
      author={Djohan Bonnet and Kellian Cottart and Tifenn Hirtzlin and Tarcisius Januel and Thomas Dalgaty and Elisa Vianello and Damien Querlioz},
      year={2025},
      eprint={2504.13569},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.13569},
}

## License

This project is licensed under the CC-BY License - see the [LICENSE](LICENSE) file for details.
