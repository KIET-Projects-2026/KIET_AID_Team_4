import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("/content/english_telugu_tts_4gb.csv", encoding='ISO-8859-1')
print("✅ Dataset loaded successfully!")
print(df.head())

df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Missing values handled.")

label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col])

print("\n✅ Categorical data encoded.")

scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n✅ Features scaled successfully.")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Dataset split into training and testing sets.")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
print("\n🎯 Preprocessing complete. Ready for modeling!")