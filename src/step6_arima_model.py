import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load training and testing data
train = pd.read_csv("output/train_data.csv")
test = pd.read_csv("output/test_data.csv")

# Convert Date column
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# Set Date as index
train.set_index("Date", inplace=True)
test.set_index("Date", inplace=True)

# Use only sales column
train_sales = train["Weekly_Sales"]
test_sales = test["Weekly_Sales"]

# Train ARIMA model (basic order)
model = ARIMA(train_sales, order=(1, 1, 1))
model_fit = model.fit()

# Forecast for test period
forecast = model_fit.forecast(steps=len(test_sales))

# Create forecast dataframe
forecast_df = test.copy()
forecast_df["ARIMA_Forecast"] = forecast.values

# Save ARIMA forecast results
forecast_df.reset_index().to_csv(
    "output/arima_forecast_results.csv",
    index=False
)

print("✅ ARIMA model training & forecasting completed")
print("Saved file: output/arima_forecast_results.csv")
print(forecast_df.head())
