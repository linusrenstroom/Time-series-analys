import numpy as np
import tensorflow as tf
from main.helpers import plot_series
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import mae

#When running this class it will take some time for the
#plot to show. It is making a lot of predictions.
#If you want to make sure the predictions aren't idle.
#Change the verbose to 1 in model.predict() to see the progression
#If you dont want the spamming in the console lines
#change the verbose to 0

model = tf.keras.models.load_model("models/model.keras")
series = np.load("data/series.npy")

split_time = 1000
window_size = 20

time = np.arange(4 * 365 + 1, dtype="float32")
time_valid = time[split_time:]
x_valid = series[split_time:]

forecast = []
for step in range(len(series) - window_size):
    forecast.append(
        model.predict(series[step:step + window_size][np.newaxis],
        verbose=1)
    )

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

mae = tf.keras.metrics.mae(x_valid, results).numpy()
print(f"MAE: {mae:.4f}")

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.legend(["Actual", "Forecast"])
plt.show()
