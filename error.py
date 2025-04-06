import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("WELfake.csv")  # Ensure the correct filename

# Check label distribution
print(df["label"].value_counts())

# Plot class distribution
sns.countplot(x=df["label"])
plt.title("Class Distribution of Fake vs Real News")
plt.show()
