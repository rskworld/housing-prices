"""
Housing Price Prediction Dataset - Data Visualization Script
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
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
print("Loading housing prices dataset for visualization...")
df = pd.read_csv('housing_prices.csv')

# Create output directory for plots
import os
os.makedirs('plots', exist_ok=True)

# 1. Price Distribution
print("Creating price distribution plot...")
plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.ticklabel_format(style='plain', axis='x')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
print("Creating correlation heatmap...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Housing Features')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Price vs Square Feet Living
print("Creating price vs square feet living plot...")
plt.figure(figsize=(10, 6))
plt.scatter(df['sqft_living'], df['price'], alpha=0.5, edgecolors='black', linewidth=0.5)
plt.xlabel('Square Feet Living')
plt.ylabel('Price ($)')
plt.title('Price vs Square Feet Living')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/price_vs_sqft_living.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Price vs Number of Bedrooms
print("Creating price vs bedrooms plot...")
plt.figure(figsize=(10, 6))
bedroom_avg_price = df.groupby('bedrooms')['price'].mean()
plt.bar(bedroom_avg_price.index, bedroom_avg_price.values, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price ($)')
plt.title('Average Price by Number of Bedrooms')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/price_vs_bedrooms.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Price vs Location (Latitude/Longitude)
print("Creating price vs location plot...")
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['long'], df['lat'], c=df['price'], 
                     cmap='viridis', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Price ($)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('House Prices by Geographic Location')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/price_by_location.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Price vs Grade
print("Creating price vs grade plot...")
plt.figure(figsize=(10, 6))
grade_avg_price = df.groupby('grade')['price'].mean()
plt.plot(grade_avg_price.index, grade_avg_price.values, marker='o', 
         linewidth=2, markersize=8, color='steelblue')
plt.fill_between(grade_avg_price.index, grade_avg_price.values, alpha=0.3)
plt.xlabel('Grade')
plt.ylabel('Average Price ($)')
plt.title('Average Price by House Grade')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/price_vs_grade.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Feature Distribution
print("Creating feature distribution plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribution of Key Features', fontsize=16, y=1.02)

features_to_plot = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'condition', 'grade']
for idx, feature in enumerate(features_to_plot):
    row = idx // 3
    col = idx % 3
    axes[row, col].hist(df[feature], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].set_title(f'Distribution of {feature}')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Price by Neighborhood (Top 10)
print("Creating price by neighborhood plot...")
neighborhood_avg = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
plt.barh(range(len(neighborhood_avg)), neighborhood_avg.values, edgecolor='black', alpha=0.7)
plt.yticks(range(len(neighborhood_avg)), neighborhood_avg.index)
plt.xlabel('Average Price ($)')
plt.title('Top 10 Neighborhoods by Average Price')
plt.ticklabel_format(style='plain', axis='x')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('plots/price_by_neighborhood.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved to 'plots' directory!")
print("Visualization complete!")

