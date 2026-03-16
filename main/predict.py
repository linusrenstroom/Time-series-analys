import numpy as np
import tensorflow as tf
from main.helpers import plot_series
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("models/model.keras")
series = np.load("series/series.npy")

split_time = 1000
window_size = 20

# Predict on the first validation window
sample = series[split_time:split_time + window_size][np.newaxis]
prediction = model.predict(sample)
print("Prediction:", prediction)

# Optional: visualize the validation portion
time = np.arange(len(series), dtype="float32")
plot_series(time, series, start=split_time)
plt.title("Validation series")
plt.show()