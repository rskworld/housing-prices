"""
Housing Price Prediction Dataset - Advanced ML Models
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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import os
from datetime import datetime

# Try importing XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

# Try importing LightGBM (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Note: LightGBM not available. Install with: pip install lightgbm")

class AdvancedHousingPricePredictor:
    """Advanced housing price prediction with multiple ML models"""
    
    def __init__(self, data_path='housing_prices.csv'):
        """Initialize the predictor with data"""
        print("Loading dataset...")
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float('inf')
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\nPreparing features...")
        
        # Select numerical features
        feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                          'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                          'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
        
        X = self.df[feature_columns].copy()
        y = self.df['price'].copy()
        
        # Feature engineering: Age of house
        current_year = 2026
        X['house_age'] = current_year - X['yr_built']
        X['renovation_age'] = X['yr_renovated'].apply(lambda x: current_year - x if x > 0 else 0)
        
        # Feature engineering: Total square feet
        X['total_sqft'] = X['sqft_above'] + X['sqft_basement']
        
        # Feature engineering: Price per square foot (for normalization)
        X['bedroom_bathroom_ratio'] = X['bedrooms'] / (X['bathrooms'] + 0.1)  # Add small value to avoid division by zero
        
        # Feature engineering: Lot ratio
        X['lot_ratio'] = X['sqft_lot'] / (X['sqft_living'] + 1)
        
        self.feature_columns = list(X.columns)
        self.X = X
        self.y = y
        
        print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        return X, y
    
    def initialize_models(self):
        """Initialize multiple ML models"""
        print("\nInitializing ML models...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=5)
        
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, max_depth=5, verbose=-1)
        
        print(f"Initialized {len(self.models)} models")
    
    def train_and_evaluate(self, test_size=0.2, cv_folds=5):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("Training and Evaluating Models")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_mape': test_mape,
                'cv_rmse': cv_rmse,
                'cv_std': cv_std
            }
            
            print(f"  Test RMSE: ${test_rmse:,.2f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MAE: ${test_mae:,.2f}")
            print(f"  CV RMSE: ${cv_rmse:,.2f} (±${cv_std:,.2f})")
            
            # Track best model
            if test_rmse < self.best_score:
                self.best_score = test_rmse
                self.best_model = name
    
    def print_summary(self):
        """Print summary of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        summary_data = []
        for name, metrics in self.results.items():
            summary_data.append({
                'Model': name,
                'Test RMSE': f"${metrics['test_rmse']:,.2f}",
                'Test R²': f"{metrics['test_r2']:.4f}",
                'Test MAE': f"${metrics['test_mae']:,.2f}",
                'Test MAPE': f"{metrics['test_mape']:.2f}%",
                'CV RMSE': f"${metrics['cv_rmse']:,.2f}",
                'CV Std': f"${metrics['cv_std']:,.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Model')
        print(summary_df.to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model}")
        print(f"Best Test RMSE: ${self.best_score:,.2f}")
        print(f"{'='*60}")
    
    def save_models(self, directory='models'):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        print(f"\nSaving models to '{directory}' directory...")
        for name, metrics in self.results.items():
            model = metrics['model']
            filename = os.path.join(directory, f"{name.lower().replace(' ', '_')}.pkl")
            joblib.dump(model, filename)
            print(f"  Saved: {filename}")
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(directory, 'feature_columns.pkl'))
        print(f"\nAll models saved successfully!")
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance for tree-based models"""
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return None
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\n{'='*60}")
            print(f"FEATURE IMPORTANCE - {model_name}")
            print(f"{'='*60}")
            print(importance_df.to_string(index=False))
            return importance_df
        else:
            print(f"{model_name} does not support feature importance")
            return None


def main():
    """Main execution function"""
    print("="*60)
    print("Advanced Housing Price Prediction Models")
    print("RSK World - Free Programming Resources & Source Code")
    print("="*60)
    
    # Initialize predictor
    predictor = AdvancedHousingPricePredictor()
    
    # Prepare features
    predictor.prepare_features()
    
    # Initialize models
    predictor.initialize_models()
    
    # Train and evaluate
    predictor.train_and_evaluate(cv_folds=5)
    
    # Print summary
    predictor.print_summary()
    
    # Feature importance
    predictor.get_feature_importance('Random Forest')
    if XGBOOST_AVAILABLE:
        predictor.get_feature_importance('XGBoost')
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

