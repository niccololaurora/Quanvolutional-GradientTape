import tensorflow as tf
import numpy as np
import csv
import optuna
import pickle
import pandas
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_rank,
    plot_slice,
    plot_timeline,
)
from optuna_objective import DetailedObjective, Objective
from help_functions import (
    initialize_data,
    validation_step,
    barplot,
    calculate_accuracy,
    validation_step,
    CNN,
    sliding,
    loss_function,
    calculate_batches,
    batch_data,
)

set_backend("tensorflow")


def main():
    # ==================================
    # Inizializzazione comune
    # ==================================
    batch_size = 10
    filterdim = 2
    resize = 10
    nclasses = 2
    number_of_trials = 2
    filt = "yes"

    # ==================================
    # Parametri specifici per la ottimizzazione di Optuna.
    # Uso train_size e epochs ridotte per favorire il Pruning.
    # ==================================
    train_size = 30
    epochs = 2
    name_barplot = "hyperparam_barplot.png"
    log_file = "hyperparam_logfile.txt"

    # Data loading and filtering
    x_train, y_train, x_test, y_test = initialize_data(train_size, resize, filt)
    barplot(name_barplot, x_train, y_train)

    # Batches
    number_of_batches, sizes_batches = calculate_batches(x_train, batch_size)

    # ==================================
    # STAMPA INFO SU FILE
    # ==================================
    with open(log_file, "a") as file:
        print(f"Labels target {y_train}", file=file)
        print(
            f"Immagini di training per Hyperparameter Optimization {len(x_train)}",
            file=file,
        )
        print(
            f"Immagini di test per Hyperparameter Optimization {len(x_test)}", file=file
        )
        print(f"Number of batches {number_of_batches}", file=file)
        print(f"Size of batch {sizes_batches}", file=file)
        print("=" * 60, file=file)

    # =================================
    # OPTUNA HYPERPARAMETER OPTIMIZATION
    # =================================
    study = optuna.create_study(
        direction="maximize", study_name="qcnn", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        Objective(
            x_train,
            y_train,
            x_test,
            y_test,
            filterdim,
            nclasses,
            epochs,
            number_of_batches,
            sizes_batches,
        ),
        n_trials=number_of_trials,
    )

    best_params = study.best_params
    best_trial = study.best_trial
    best_trials = study.best_trials

    with open("best_hyper_optuna.txt", "a") as file:
        print("=" * 60, file=file)
        print("=" * 60, file=file)
        print(f"best_params {best_params}", file=file)
        print("=" * 60, file=file)
        print("=" * 60, file=file)
        print(f"best_trial {best_trial}", file=file)
        print("=" * 60, file=file)
        print("=" * 60, file=file)
        print(f"best_trials {best_trials}", file=file)
        print("=" * 60, file=file)
        print("=" * 60, file=file)

    # Plot di ottimizzazione della storia
    fig = plot_optimization_history(study)
    fig.write_image("optimization_history.png")

    # Plot dei valori intermedi
    fig = plot_intermediate_values(study)
    fig.write_image("intermediate_values.png")

    # Plot del contorno
    fig = plot_contour(study)
    fig.write_image("contour_plot.png")

    # Plot di slice
    fig = plot_slice(study)
    fig.write_image("slice_plot.png")

    # Plot delle importanze dei parametri
    fig = plot_param_importances(study)
    fig.write_image("param_importances.png")

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # =================================
    # OPTUNA BEST HYPERPARAMETER
    # =================================
    log_file = "best_trial.txt"
    name_barplot = "best_trial_barplot.png"
    train_size = 30
    batch_size = 10
    epochs = 2
    x_train, y_train, x_test, y_test = initialize_data(train_size, resize, filt)
    barplot(name_barplot, x_train, y_train)
    number_of_batches, sizes_batches = calculate_batches(x_train, batch_size)

    # ==================================
    # STAMPA INFO SU FILE
    # ==================================
    with open(log_file, "a") as file:
        print(f"Parametri iniziale circuito {vparams}", file=file)
        print(f"Labels target {y_train}", file=file)
        print(
            f"Immagini di training per Hyperparameter Optimization {len(x_train)}",
            file=file,
        )
        print(
            f"Immagini di test per Hyperparameter Optimization {len(x_test)}", file=file
        )
        print(f"Number of batches {number_of_batches}", file=file)
        print(f"Size of batch {sizes_batches}", file=file)
        print("=" * 60, file=file)

    detailed_objective = DetailedObjective(
        x_train,
        y_train,
        x_test,
        y_test,
        filterdim,
        nclasses,
        epochs,
        number_of_batches,
        sizes_batches,
        log_file,
    )
    detailed_objective(study.best_trial)


if __name__ == "__main__":
    main()
