"""
Housing Price Prediction Dataset - Data Analysis Script
RSK World - Free Programming Resources & Source Code
Website: https://rskworld.in
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Founder: Molla Samser
Designer & Tester: Rima Khatun
Created: 2026
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading housing prices dataset...")
df = pd.read_csv('housing_prices.csv')

# Display basic information about the dataset
print("\n" + "="*50)
print("Dataset Overview")
print("="*50)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "="*50)
print("Dataset Statistics")
print("="*50)
print(df.describe())

print("\n" + "="*50)
print("Data Types and Missing Values")
print("="*50)
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")

# Basic feature analysis
print("\n" + "="*50)
print("Price Statistics")
print("="*50)
print(f"Mean Price: ${df['price'].mean():,.2f}")
print(f"Median Price: ${df['price'].median():,.2f}")
print(f"Min Price: ${df['price'].min():,.2f}")
print(f"Max Price: ${df['price'].max():,.2f}")
print(f"Standard Deviation: ${df['price'].std():,.2f}")

# Correlation analysis
print("\n" + "="*50)
print("Correlation with Price")
print("="*50)
correlations = df.select_dtypes(include=[np.number]).corr()['price'].sort_values(ascending=False)
print(correlations)

# Prepare features for modeling
print("\n" + "="*50)
print("Preparing Model Features")
print("="*50)

# Select numerical features for the model
feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                   'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                   'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

X = df[feature_columns]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train a simple linear regression model
print("\n" + "="*50)
print("Training Linear Regression Model")
print("="*50)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Training RMSE: ${train_rmse:,.2f}")
print(f"Test RMSE: ${test_rmse:,.2f}")

# Feature importance (coefficients)
print("\n" + "="*50)
print("Feature Coefficients")
print("="*50)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False, key=abs)

print(feature_importance)

print("\n" + "="*50)
print("Analysis Complete!")
print("="*50)

