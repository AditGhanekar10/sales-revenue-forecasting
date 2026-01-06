import pandas as pd

# Load weekly aggregated sales data
df = pd.read_csv("output/weekly_sales_forecast_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Create a 4-week moving average baseline forecast
df["Moving_Avg_4W"] = df["Weekly_Sales"].rolling(window=4).mean()

# Drop initial rows with NaN moving average
df = df.dropna().reset_index(drop=True)

# Save baseline forecast
df.to_csv(
    "output/baseline_moving_average_forecast.csv",
    index=False
)

print("✅ Baseline forecasting completed")
print("Saved file: output/baseline_moving_average_forecast.csv")
print(df.head())
