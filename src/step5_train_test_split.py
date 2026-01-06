import pandas as pd

# Load baseline forecast data
df = pd.read_csv("output/baseline_moving_average_forecast.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Define train-test split (80% train, 20% test)
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

# Save train and test sets
train.to_csv("output/train_data.csv", index=False)
test.to_csv("output/test_data.csv", index=False)

print("✅ Train-test split completed")
print("Train rows:", len(train))
print("Test rows:", len(test))
