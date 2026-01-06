import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Load data
actual = pd.read_csv("output/weekly_sales_forecast_data.csv")
arima = pd.read_csv("output/arima_forecast_results.csv")
xgb = pd.read_csv("output/xgboost_forecast_results.csv")
lstm = pd.read_csv("output/lstm_forecast_results.csv")

# Convert Date columns
actual["Date"] = pd.to_datetime(actual["Date"])
arima["Date"] = pd.to_datetime(arima["Date"])
xgb["Date"] = pd.to_datetime(xgb["Date"])
lstm["Date"] = pd.to_datetime(lstm["Date"])

# Create plot
plt.figure(figsize=(14, 7))

plt.plot(actual["Date"], actual["Weekly_Sales"], label="Actual Sales", linewidth=2)
plt.plot(arima["Date"], arima["ARIMA_Forecast"], label="ARIMA Forecast", linestyle="--")
plt.plot(xgb["Date"], xgb["XGBoost_Forecast"], label="XGBoost Forecast", linestyle="--")
plt.plot(lstm["Date"], lstm["LSTM_Forecast"], label="LSTM Forecast", linestyle="--")

# Titles and labels
plt.title("Weekly Sales Forecast Comparison", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()

# Save plot
plt.tight_layout()
plt.savefig("output/sales_forecast_comparison.png", dpi=300)
plt.show()

print("✅ Plot saved as output/sales_forecast_comparison.png")
