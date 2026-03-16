import tensorflow as tf
import numpy as np

from keras_tuner import RandomSearch
from main.helpers import trend, seasonality, noise, windowed_dataset

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


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units', min_value=10, max_value=30, step=2),
        activation="relu", input_shape=[window_size]))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=1e-5,
        momentum=hp.Choice(
            "momentum",
            values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
    )

    model.compile(loss="mse", optimizer=optimizer)

    return model


tuner = RandomSearch(
    build_model,
    objective='loss',
    max_trials=20,
    executions_per_trial=2,
    directory='my_dir',
    project_name='hello',
    overwrite=True
)

tuner.search(dataset,
             epochs=100,
             verbose=1,
             steps_per_epoch=len(x_train) // batch_size)

print("Search finished")

tuner.results_summary()

tuner.get_best_models(num_models=1)

best_hp = tuner.get_best_hyperparameters(1)[0]

print(best_hp.values)
