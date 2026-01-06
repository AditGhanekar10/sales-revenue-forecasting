import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load ARIMA forecast results
df = pd.read_csv("output/arima_forecast_results.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Actual vs predicted values
y_true = df["Weekly_Sales"]
y_pred = df["ARIMA_Forecast"]

# Calculate evaluation metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Create evaluation summary
evaluation_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Value": [mae, rmse]
})

# Save evaluation results
evaluation_df.to_csv(
    "output/arima_model_evaluation.csv",
    index=False
)

print("✅ Model evaluation completed")
print("\nEvaluation Metrics:")
print(evaluation_df)
