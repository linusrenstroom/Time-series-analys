A neural network model that learns to forecast a synthetic time series built from trend, seasonality, and noise components.
This project generates a synthetic time series and trains a simple dense neural network to predict future values using a sliding window approach.
It's a practical introduction to time series forecasting with Keras



├── utils.py       # Helper functions (data generation, windowing, plotting)
├── train.py       # Data prep, model definition, and training
├── predict.py     # Load saved model and run predictions
├── model.keras    # Saved model (generated after training)
└── series.npy     # Saved series data (generated after training)
