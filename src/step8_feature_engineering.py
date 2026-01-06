import pandas as pd

# Load weekly aggregated data
df = pd.read_csv("output/weekly_sales_forecast_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Create lag features
df["lag_1"] = df["Weekly_Sales"].shift(1)
df["lag_2"] = df["Weekly_Sales"].shift(2)
df["lag_4"] = df["Weekly_Sales"].shift(4)

# Create rolling mean features
df["rolling_mean_4"] = df["Weekly_Sales"].rolling(window=4).mean()
df["rolling_mean_12"] = df["Weekly_Sales"].rolling(window=12).mean()

# Drop rows with missing values
df = df.dropna().reset_index(drop=True)

# Save feature-engineered data
df.to_csv(
    "output/ml_feature_data.csv",
    index=False
)

print("✅ Feature engineering completed")
print("Saved file: output/ml_feature_data.csv")
print(df.head())
