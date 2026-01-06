import pandas as pd
import os

# Print current working directory (for debugging)
print("Current working directory:")
print(os.getcwd())

# Load dataset using absolute-safe path
df = pd.read_csv("data/walmart_sales.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATA INFO")
print(df.info())

print("\nBASIC STATS")
print(df.describe())
