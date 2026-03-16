import numpy as np
import tensorflow as tf
from tensorflow import keras
from helpers import trend, seasonality, noise, windowed_dataset

# Data
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

series = baseline + trend(time, slope)
series += seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
)

model.fit(dataset, epochs=100, verbose=1)
model.save("model.keras")

np.save("series/series.npy", series)
print("Training complete. Model saved to model.keras")