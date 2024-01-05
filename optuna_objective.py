import tensorflow as tf
import numpy as np
import csv
import optuna
import pickle
import pandas
import cProfile
import pstats
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


class Objective:
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        filterdim,
        nclasses,
        epochs,
        number_of_batches,
        size_batches,
    ):
        self.singleQ = {
            "X-rotation": "RX",
            "Y-rotation": "RY",
            "Z-rotation": "RZ",
            "Generic Unitary": "U3",
            "Phase gate": "S",
            "T-gate": "T",
            "Hadamard": "H",
        }
        self.twoQ = {
            "Cnot": "CNOT",
            "Swap": "SWAP",
            "sqrtSwap": "GeneralizedfSim",
            "ControlledU": "CU3",
            "ControlledRX": "CRX",
            "ControlledRY": "CRY",
            "ControlledRZ": "CRZ",
        }
        self.filterdim = filterdim
        self.nclasses = nclasses
        self.epochs = epochs
        self.number_of_batches = number_of_batches
        self.sizes_batches = size_batches
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.vparams = tf.Variable(tf.random.normal(shape=(23,)), trainable=True)

    def __call__(self, trial):
        # Profiler
        profiler = cProfile.Profile()
        profiler.enable()

        # ==========================
        # OPTUNA
        # ==========================
        exp = trial.suggest_int("lr", 1, 5)
        lr = 10 ** (-exp)
        n_kernels_1 = trial.suggest_int("n_kernels_1", 4, 128, log=False)
        n_kernels_2 = trial.suggest_int("n_kernels_2", 4, 128, log=False)
        n_nodes = trial.suggest_int("n_nodes", 10, 100, log=False)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        cnn = CNN(trial, self.nclasses, n_kernels_1, n_kernels_2, n_nodes)

        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        for i in range(self.epochs):
            # ==================================
            # BATCH
            # ==================================
            loss_batch = []
            acc_batch = []
            for k in range(self.number_of_batches):
                # Seleziono i dati di training in base alla batch
                x, y = batch_data(
                    k,
                    self.x_train,
                    self.y_train,
                    self.number_of_batches,
                    self.sizes_batches,
                )

                # GRADIENT BATCH
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(self.vparams)
                    l_batch = loss_function(x, y, self.vparams, cnn, self.filterdim)

                grad_cnn = tape.gradient(l_batch, cnn.trainable_weights)
                grad_qcnn = tape.gradient(l_batch, [self.vparams])
                optimizer.apply_gradients(zip(grad_cnn, cnn.trainable_weights))
                optimizer.apply_gradients(zip(grad_qcnn, [self.vparams]))

                # ACCURACY BATCH
                predictions = []
                for image in x:
                    output_qfilter = sliding(image, self.vparams, self.filterdim)
                    prediction = cnn(output_qfilter)
                    predictions.append(prediction)
                accuracy_batch = calculate_accuracy(predictions, y)

                # APPEND ACCURACY AND LOSS
                acc_batch.append(accuracy_batch)
                loss_batch.append(l_batch)

            # TRAINING: loss and accuracy
            train_a_epoch = tf.reduce_sum(acc_batch) / self.number_of_batches
            train_l_epoch = tf.reduce_sum(loss_batch) / self.number_of_batches
            train_acc.append(train_a_epoch)
            train_loss.append(train_l_epoch)

            # VALIDATION: loss and accuracy
            val_l_epoch, val_a_epoch = validation_step(
                self.x_test, self.y_test, self.vparams, cnn
            )
            val_loss.append(val_l_epoch)
            val_acc.append(val_a_epoch)

            # PRUNING OPTUNA: I use as pruning metric the validation accuracy epoch value
            trial.report(val_a_epoch, i)

            if trial.should_prune():
                raise optuna.TrialPruned()

        profiler.disable()
        with open("profile_results_objective.txt", "w") as profile_file:
            stats = pstats.Stats(profiler, stream=profile_file)
            stats.sort_stats("cumulative")
            stats.print_stats()
        return tf.reduce_sum(val_acc)


class DetailedObjective:
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        filterdim,
        nclasses,
        epochs,
        number_of_batches,
        size_batches,
        log_file,
    ):
        self.singleQ = {
            "X-rotation": "RX",
            "Y-rotation": "RY",
            "Z-rotation": "RZ",
            "Generic Unitary": "U3",
            "Phase gate": "S",
            "T-gate": "T",
            "Hadamard": "H",
        }
        self.twoQ = {
            "Cnot": "CNOT",
            "Swap": "SWAP",
            "sqrtSwap": "GeneralizedfSim",
            "ControlledU": "CU3",
            "ControlledRX": "CRX",
            "ControlledRY": "CRY",
            "ControlledRZ": "CRZ",
        }
        self.filterdim = filterdim
        self.nclasses = nclasses
        self.epochs = epochs
        self.log_file = log_file
        self.number_of_batches = number_of_batches
        self.sizes_batches = size_batches
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.vparams = tf.Variable(tf.random.normal(shape=(23,)), trainable=True)

    def __call__(self, trial):
        # Profiler
        profiler = cProfile.Profile()
        profiler.enable()
        # ==========================
        # OPTUNA
        # ==========================
        exp = trial.suggest_int("lr", 1, 5)
        lr = 10 ** (-exp)
        n_kernels_1 = trial.suggest_int("n_kernels_1", 4, 128, log=False)
        n_kernels_2 = trial.suggest_int("n_kernels_2", 4, 128, log=False)
        n_nodes = trial.suggest_int("n_nodes", 10, 100, log=False)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        cnn = CNN(trial, self.nclasses, n_kernels_1, n_kernels_2, n_nodes)

        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        for i in range(self.epochs):
            with open(self.log_file, "a") as file:
                print("=" * 60, file=file)
                print(f"Epoch {i+1}", file=file)

            # ==================================
            # BATCH
            # ==================================
            loss_batch = []
            acc_batch = []
            for k in range(self.number_of_batches):
                # Seleziono i dati di training in base alla batch
                x, y = batch_data(
                    k,
                    self.x_train,
                    self.y_train,
                    self.number_of_batches,
                    self.sizes_batches,
                )

                # GRADIENT BATCH
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(self.vparams)
                    l_batch = loss_function(x, y, self.vparams, cnn, self.filterdim)

                grad_cnn = tape.gradient(l_batch, cnn.trainable_weights)
                grad_qcnn = tape.gradient(l_batch, [self.vparams])
                optimizer.apply_gradients(zip(grad_cnn, cnn.trainable_weights))
                optimizer.apply_gradients(zip(grad_qcnn, [self.vparams]))

                # ACCURACY BATCH
                predictions = []
                for image in x:
                    output_qfilter = sliding(image, self.vparams, self.filterdim)
                    prediction = cnn(output_qfilter)
                    predictions.append(prediction)
                accuracy_batch = calculate_accuracy(predictions, y)

                # APPEND ACCURACY AND LOSS
                acc_batch.append(accuracy_batch)
                loss_batch.append(l_batch)

                with open(self.log_file, "a") as file:
                    print("=" * 60, file=file)
                    print(f"Batch {k+1}", file=file)
                    print(f"Loss Batch {k+1}: {l_batch}", file=file)
                    print(f"Acc Batch {k+1}: {accuracy_batch}", file=file)
                    print(f"Gradients {grad_qcnn}", file=file)
                    print(f"Circuit params {self.vparams}", file=file)

            # TRAINING: loss and accuracy
            train_a_epoch = tf.reduce_sum(acc_batch) / self.number_of_batches
            train_l_epoch = tf.reduce_sum(loss_batch) / self.number_of_batches
            train_acc.append(train_a_epoch)
            train_loss.append(train_l_epoch)

            # VALIDATION: loss and accuracy
            val_l_epoch, val_a_epoch = validation_step(
                self.x_test, self.y_test, self.vparams, cnn
            )
            val_loss.append(val_l_epoch)
            val_acc.append(val_a_epoch)

            # ==========================
            # Salvo:
            # 1. Pesi della CNN
            # 2. Parametri circuito
            # 3. Metriche della epoca
            # ==========================
            cnn.save_weights("saved_weights.h5")

            with open("saved_parameters.pkl", "wb") as f:
                pickle.dump(self.vparams, f, pickle.HIGHEST_PROTOCOL)

            with open("epoch.csv", "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(
                    ["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc"]
                )
                csv_writer.writerow(
                    [
                        i + 1,
                        train_l_epoch.numpy(),
                        val_l_epoch.numpy(),
                        train_a_epoch.numpy(),
                        val_a_epoch.numpy(),
                    ]
                )

            with open(self.log_file, "a") as file:
                print("/" * 60, file=file)
                print(f"Train loss Epoch {i+1}: {train_l_epoch}", file=file)
                print(f"Val loss Epoch {i+1}: {val_l_epoch}", file=file)
                print("/" * 60, file=file)

        with open(self.log_file, "a") as file:
            print("/" * 60, file=file)
            print("/" * 60, file=file)
            print("Fine del training", file=file)
            print("/" * 60, file=file)
            print("/" * 60, file=file)

        profiler.disable()
        with open("profile_results_detailed_objective.txt", "w") as profile_file:
            stats = pstats.Stats(profiler, stream=profile_file)
            stats.sort_stats("cumulative")
            stats.print_stats()

        return tf.reduce_sum(val_acc)
