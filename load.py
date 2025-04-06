import pandas as pd

# Load dataset
df = pd.read_csv("WELFake.csv")

# Display first few rows
print(df.head())

# Check label distribution
print(df['label'].value_counts())
