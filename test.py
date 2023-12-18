import tensorflow as tf

predictions = [0.8, 0.2, 0.3]
labels = [1, 0, 1]

accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.9)
accuracy.update_state(labels, predictions)

result = accuracy.result().numpy()

print(f"Accuracy: {result}")
