# Quanvolutional-GradientTape

Implementing the Quanvolutional NN described in [arXiv:1904.04767](https://arxiv.org/abs/1904.04767/) with [Qibo](https://qibo.science/)

This is the same architecture implemented in the repository Quanvolutional_NN_API with a different approach.
The Quanvolutional_NN_API implementation exploits the high level APIs of TensorFlow: `model.compile`, `model.fit`, `model.predict`.
In this repository, I do not use TensorFlow's APIs; instead, I employ a custom loop based on the GradientTape context manager.

The optimization of the architecture's hyperparameters has been run using [Optuna](https://optuna.org/).
Moreover I implemented the profiling of the code using `cProfile`.

Files:
1. `cluster.py` is the main file necessary to run the training of the Quanvolutional Neural Network.
2. `ansatz.py` contains Qibo's circuits used as ansatze for the Quanvolutional filters of the Quanvolutional Neural Network.
3. `help_functions.py` contains helpful functions.
4. `optuna_objective.py` contains two classes: Objective and DetailedObjective. The first class is used to run the various trials of Optuna and identify the best set of hyperparameters 
    (average pruning is adopted to prune unpromising trials). The second class is used to run the complete training of the best trial (best set of hyperparameters) found by Optuna.
5. `random_circuit_generator.py` contains some functions to build a pseudo-random Qibo's circuit, which could be used as Quanvolutional filters. The recipe to build the pseudo-random 
    Qibo's circuit is the one suggested in the paper.
6. `init.py` should be run before running the training to modify the training specifics contained in `cluster.py`.
7. `config_file.py` is needed to change the name of the training job in a GPU-cluster managed by [Slurm](https://slurm.schedmd.com/documentation.html)
