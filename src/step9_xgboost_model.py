import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load ML feature data
df = pd.read_csv("output/ml_feature_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Features and target
X = df.drop(columns=["Date", "Weekly_Sales"])
y = df["Weekly_Sales"]

# Time-series train-test split (80/20)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Train XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save predictions
results = df.iloc[split_index:].copy()
results["XGBoost_Forecast"] = y_pred

results.to_csv(
    "output/xgboost_forecast_results.csv",
    index=False
)

# Save evaluation metrics
evaluation_df = pd.DataFrame({
    "Model": ["XGBoost"],
    "MAE": [mae],
    "RMSE": [rmse]
})

evaluation_df.to_csv(
    "output/xgboost_model_evaluation.csv",
    index=False
)

print("✅ XGBoost model training & forecasting completed")
print("\nEvaluation Metrics:")
print(evaluation_df)
