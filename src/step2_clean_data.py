import pandas as pd

# Load raw data
df = pd.read_csv("data/walmart_sales.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Select only useful columns
df = df[
    ["Date", "Store", "Dept", "Weekly_Sales", "IsHoliday"]
]

# Remove rows with negative sales (invalid)
df = df[df["Weekly_Sales"] >= 0]

# Sort data by date
df = df.sort_values("Date")

# Reset index
df = df.reset_index(drop=True)

# Save cleaned data
df.to_csv("output/cleaned_sales_data.csv", index=False)

print("✅ Data cleaning completed")
print("Rows after cleaning:", len(df))
print("Saved file: output/cleaned_sales_data.csv")
