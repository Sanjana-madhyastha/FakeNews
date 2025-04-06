import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("WELFake.csv")

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Drop missing values in the "text" column
df = df.dropna(subset=["text"])

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):  # Ensure text is a string
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = text.split()  # Split into words
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return " ".join(tokens)

# Apply text cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Display cleaned data
print(df[["clean_text", "label"]].head())

# Save cleaned data
df.to_csv("cleaned_fakenews.csv", index=False)
