import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load weekly sales data
df = pd.read_csv("output/weekly_sales_forecast_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Use only sales column
sales = df["Weekly_Sales"].values.reshape(-1, 1)

# Scale data (required for LSTM)
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales)

# Create sequences
def create_sequences(data, window_size=12):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(sales_scaled, window_size=12)

# Train-test split (80/20)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# Train model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    verbose=1
)

# Predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Evaluation
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

# Save results
results = df.iloc[-len(y_pred):].copy()
results["LSTM_Forecast"] = y_pred.flatten()

results.to_csv(
    "output/lstm_forecast_results.csv",
    index=False
)

evaluation_df = pd.DataFrame({
    "Model": ["LSTM"],
    "MAE": [mae],
    "RMSE": [rmse]
})

evaluation_df.to_csv(
    "output/lstm_model_evaluation.csv",
    index=False
)

print("✅ LSTM model training & forecasting completed")
print("\nEvaluation Metrics:")
print(evaluation_df)
