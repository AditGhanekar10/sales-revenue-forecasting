import pandas as pd

# Load cleaned data
df = pd.read_csv("output/cleaned_sales_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Aggregate total sales per week
weekly_sales = (
    df.groupby("Date", as_index=False)
      .agg({"Weekly_Sales": "sum"})
)

# Sort by date
weekly_sales = weekly_sales.sort_values("Date")

# Save aggregated data
weekly_sales.to_csv(
    "output/weekly_sales_forecast_data.csv",
    index=False
)

print("✅ Weekly aggregation completed")
print("Total weeks:", len(weekly_sales))
print("Saved file: output/weekly_sales_forecast_data.csv")
