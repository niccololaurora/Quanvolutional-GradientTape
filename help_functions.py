import tensorflow as tf
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from ansatz import (
    SimplifiedTwoDesign,
    BasicEntanglerLayers,
    Havlicek,
    Mine1,
    Mine2,
    Mine3,
    Mine4,
)
from qibo import set_backend, gates, Circuit

set_backend("tensorflow")


def batch_data(k, x_train, y_train, number_of_batches, sizes_batches):
    if k == number_of_batches - 1:
        x = x_train[
            sizes_batches[k - 1] * k : sizes_batches[k - 1] * k + sizes_batches[k]
        ]
        y = y_train[
            sizes_batches[k - 1] * k : sizes_batches[k - 1] * k + sizes_batches[k]
        ]
    else:
        x = x_train[sizes_batches[k] * k : sizes_batches[k] * (k + 1)]
        y = y_train[sizes_batches[k] * k : sizes_batches[k] * (k + 1)]

    return x, y


def calculate_batches(x_train, batch_size):
    if len(x_train) % batch_size == 0:
        number_of_batches = int(len(x_train) / batch_size)
        sizes_batches = [batch_size for i in range(number_of_batches)]
    else:
        number_of_batches = int(len(x_train) / batch_size) + 1
        size_last_batch = len(x_train) - batch_size * int(len(x_train) / batch_size)
        sizes_batches = [batch_size for i in range(number_of_batches - 1)]
        sizes_batches.append(size_last_batch)

    return number_of_batches, sizes_batches


def sliding(image, vparams, filterdim):
    shots = 10
    image_heigth, image_width = 10, 10
    q_image_width = image_width - filterdim + 1
    q_image_heigth = image_heigth - filterdim + 1
    q_image_1 = []
    q_image_2 = []
    quanvolutional_filter1 = Mine3()  # 12
    quanvolutional_filter2 = Mine4()  # 11

    param1 = vparams[0:12]
    param2 = vparams[12:23]

    quanvolutional_filter1.set_parameters(param1)
    quanvolutional_filter2.set_parameters(param2)

    for i in range(q_image_width):
        for j in range(q_image_width):
            # ================
            # Encoding
            # ================
            roi = image[i : i + filterdim, j : j + filterdim]
            flattened_roi = tf.reshape(roi, [-1])
            encoding_circuit = Circuit(4, density_matrix=True)
            for k, item in enumerate(flattened_roi):
                if item >= 0.5:
                    encoding_circuit.add(gates.X(k))
            result = encoding_circuit()
            state = result.state()

            # ================
            # Decoding
            # ================
            result1 = quanvolutional_filter1(state)
            result2 = quanvolutional_filter2(state)
            pixel_1 = 0
            pixel_2 = 0
            for q in range(4):
                pixel_1 += result1.probabilities(qubits=[q])[1]
                pixel_2 += result2.probabilities(qubits=[q])[1]

            q_image_1.append(pixel_1)
            q_image_2.append(pixel_2)

    q_image_1 = tf.cast(q_image_1, dtype=tf.float32)
    q_image_2 = tf.cast(q_image_2, dtype=tf.float32)

    q_image_1 = tf.reshape(q_image_1, [1, q_image_width, q_image_heigth])
    q_image_2 = tf.reshape(q_image_2, [1, q_image_width, q_image_heigth])

    q_image = tf.stack([q_image_1, q_image_2], axis=-1)

    return q_image


def loss_function(x_train, y_train, vparams, cnn, filterdim):
    counter = 0
    loss = 0
    predictions = []
    for x, y in zip(x_train, y_train):
        print(f"Immagine {counter + 1}")
        output_qfilter = sliding(x, vparams, filterdim)
        output_qcnn = cnn(output_qfilter)
        output_qcnn = tf.reshape(output_qcnn, [-1])
        predictions.append(output_qcnn)
        counter += 1

    return tf.keras.losses.BinaryCrossentropy()(y_train, predictions)


def validation_step(x_test, y_test, vparams, cnn):
    predictions = []
    loss = 0
    for x, y in zip(x_test, y_test):
        output_qfilter = sliding(x, vparams, 2)
        output_qcnn = cnn(output_qfilter)
        output_qcnn = tf.reshape(output_qcnn, [-1])
        predictions.append(output_qcnn)

    y_test = tf.cast(y_test, dtype=tf.float32)
    predictions = tf.cast(predictions, dtype=tf.float32)
    loss = tf.keras.losses.BinaryCrossentropy()(predictions, y_test)
    accuracy = calculate_accuracy(predictions, y_test)
    return loss, accuracy


def CNN(trial, nclasses, n_kernels_1, n_kernels_2, n_nodes):
    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Conv2D(n_kernels_1, (2, 2), activation="relu"))
    cnn.add(tf.keras.layers.AveragePooling2D())
    cnn.add(tf.keras.layers.Conv2D(n_kernels_2, (2, 2), activation="relu"))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(n_nodes, activation="relu"))
    if nclasses == 2:
        cnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    else:
        cnn.add(tf.keras.layers.Dense(nclasses, activation="softmax"))

    return cnn


def calculate_accuracy(predictions, labels):
    accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.6)
    accuracy.update_state(labels, predictions)

    with open("accuracy.txt", "a") as file:
        print("/" * 60, file=file)
        print(f"Predictions: {predictions}", file=file)
        print(f"Labels: {labels}", file=file)
        print(f"Accuracy {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)

    return accuracy.result().numpy()


def plot_metrics(name_file, train_loss, val_loss, train_acc, val_acc, epochs):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, val_loss, label="validation")
    ax[0].set_title("QModel Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label="train")
    ax[1].plot(epochs, val_acc, label="validation")
    ax[1].set_title("QModel Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].legend()

    plt.savefig(name_file)
    plt.show()


def barplot(name, x_train, y_train):
    mask_0 = y_train == 0
    mask_1 = y_train == 1
    x_0 = x_train[mask_0]
    x_1 = x_train[mask_1]

    digits = {"0": len(x_0), "1": len(x_1)}
    plt.bar(digits.keys(), digits.values(), color="maroon", width=0.4)
    plt.xlabel("Digits")
    plt.title(f"Occurences of 0, 1")
    plt.savefig(name)


def initialize_data(train_size, resize, filt):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if filt == "yes":
        mask_train = (y_train == 0) | (y_train == 1)
        mask_test = (y_test == 0) | (y_test == 1)
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]
        x_test = x_test[mask_test]
        y_test = y_test[mask_test]

    if train_size != 0:
        x_train = x_train[0:train_size]
        y_train = y_train[0:train_size]
        x_test = x_test[train_size + 1 : (train_size + 1) * 2]
        y_test = y_test[train_size + 1 : (train_size + 1) * 2]

    # Resize images
    width, length = 10, 10

    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    x_train = tf.image.resize(x_train, [width, length])
    x_test = tf.image.resize(x_test, [width, length])

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()

    return x_train, y_train, x_test, y_test
