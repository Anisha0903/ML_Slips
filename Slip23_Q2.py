import pandas as pd

df = pd.read_csv("your_file.csv")

print("Null values in each column:")
print(df.isnull().sum())

df_clean = df.dropna()

print("\nAfter removing null values:")
print(df_clean.isnull().sum())
