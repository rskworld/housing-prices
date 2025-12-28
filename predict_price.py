"""
Housing Price Prediction Dataset - Price Prediction Script
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
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

class HousingPricePredictor:
    """Predict housing prices using trained models"""
    
    def __init__(self, model_path=None):
        """Initialize predictor with trained model"""
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded model from: {model_path}")
        else:
            # Train a default model if no model provided
            print("Training default model...")
            self.train_default_model()
    
    def train_default_model(self):
        """Train a default Random Forest model"""
        df = pd.read_csv('housing_prices.csv')
        
        feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                          'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                          'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
        
        current_year = 2026
        X = df[feature_columns].copy()
        X['house_age'] = current_year - X['yr_built']
        X['total_sqft'] = X['sqft_above'] + X['sqft_basement']
        
        y = df['price']
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        self.feature_columns = list(X.columns)
        print("Default model trained successfully!")
    
    def prepare_features(self, property_data):
        """Prepare features from property data dictionary"""
        current_year = 2026
        
        # Create feature array
        features = np.array([[
            property_data.get('bedrooms', 3),
            property_data.get('bathrooms', 2),
            property_data.get('sqft_living', 2000),
            property_data.get('sqft_lot', 8000),
            property_data.get('floors', 1),
            property_data.get('waterfront', 0),
            property_data.get('view', 0),
            property_data.get('condition', 3),
            property_data.get('grade', 7),
            property_data.get('sqft_above', 2000),
            property_data.get('sqft_basement', 0),
            property_data.get('yr_built', 2000),
            property_data.get('yr_renovated', 0),
            property_data.get('sqft_living15', 2000),
            property_data.get('sqft_lot15', 8000)
        ]])
        
        # Add engineered features
        house_age = current_year - features[0, 11]  # yr_built
        total_sqft = features[0, 9] + features[0, 10]  # sqft_above + sqft_basement
        
        features = np.append(features[0], [house_age, total_sqft])
        return features.reshape(1, -1)
    
    def predict(self, property_data):
        """Predict price for a property"""
        features = self.prepare_features(property_data)
        prediction = self.model.predict(features)[0]
        return prediction
    
    def predict_batch(self, properties_list):
        """Predict prices for multiple properties"""
        predictions = []
        for prop in properties_list:
            pred = self.predict(prop)
            predictions.append(pred)
        return predictions


def example_usage():
    """Example usage of the predictor"""
    print("="*60)
    print("Housing Price Prediction Example")
    print("RSK World - Free Programming Resources & Source Code")
    print("="*60)
    
    # Initialize predictor
    predictor = HousingPricePredictor()
    
    # Example property
    example_property = {
        'bedrooms': 3,
        'bathrooms': 2,
        'sqft_living': 2000,
        'sqft_lot': 8000,
        'floors': 2,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'grade': 7,
        'sqft_above': 2000,
        'sqft_basement': 0,
        'yr_built': 2000,
        'yr_renovated': 0,
        'sqft_living15': 2000,
        'sqft_lot15': 8000
    }
    
    # Predict price
    predicted_price = predictor.predict(example_property)
    
    print(f"\nExample Property:")
    for key, value in example_property.items():
        print(f"  {key}: {value}")
    
    print(f"\nPredicted Price: ${predicted_price:,.2f}")
    print("\n" + "="*60)


if __name__ == "__main__":
    example_usage()

