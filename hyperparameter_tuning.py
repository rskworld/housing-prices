"""
Housing Price Prediction Dataset - Hyperparameter Tuning
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
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import joblib
import os

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class HyperparameterTuner:
    """Hyperparameter tuning for housing price prediction models"""
    
    def __init__(self, data_path='housing_prices.csv'):
        """Initialize tuner"""
        print("Loading dataset...")
        self.df = pd.read_csv(data_path)
        self.best_params = {}
        self.best_models = {}
        
    def prepare_data(self):
        """Prepare data for tuning"""
        print("\nPreparing data...")
        
        feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                          'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                          'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
        
        # Add engineered features
        current_year = 2026
        X = self.df[feature_columns].copy()
        X['house_age'] = current_year - X['yr_built']
        X['total_sqft'] = X['sqft_above'] + X['sqft_basement']
        
        y = self.df['price']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def tune_random_forest(self, method='grid', n_iter=50):
        """Tune Random Forest hyperparameters"""
        print("\n" + "="*60)
        print("Tuning Random Forest Hyperparameters")
        print("="*60)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        # Scoring metric
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        if method == 'grid':
            search = GridSearchCV(
                rf, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                rf, param_grid, n_iter=n_iter, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1,
                random_state=42, verbose=1
            )
        
        print("Searching for best parameters...")
        search.fit(self.X_train, self.y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        # Evaluate
        y_pred = best_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"\nBest Parameters: {best_params}")
        print(f"Best CV Score (RMSE): ${np.sqrt(-search.best_score_):,.2f}")
        print(f"Test RMSE: ${rmse:,.2f}")
        print(f"Test R²: {r2:.4f}")
        
        self.best_models['Random Forest'] = best_model
        self.best_params['Random Forest'] = best_params
        
        return best_model, best_params
    
    def tune_gradient_boosting(self, method='grid', n_iter=50):
        """Tune Gradient Boosting hyperparameters"""
        print("\n" + "="*60)
        print("Tuning Gradient Boosting Hyperparameters")
        print("="*60)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        
        if method == 'grid':
            search = GridSearchCV(
                gb, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                gb, param_grid, n_iter=n_iter, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1,
                random_state=42, verbose=1
            )
        
        print("Searching for best parameters...")
        search.fit(self.X_train, self.y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        # Evaluate
        y_pred = best_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"\nBest Parameters: {best_params}")
        print(f"Best CV Score (RMSE): ${np.sqrt(-search.best_score_):,.2f}")
        print(f"Test RMSE: ${rmse:,.2f}")
        print(f"Test R²: {r2:.4f}")
        
        self.best_models['Gradient Boosting'] = best_model
        self.best_params['Gradient Boosting'] = best_params
        
        return best_model, best_params
    
    def tune_xgboost(self, method='grid', n_iter=50):
        """Tune XGBoost hyperparameters"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Skipping...")
            return None, None
        
        print("\n" + "="*60)
        print("Tuning XGBoost Hyperparameters")
        print("="*60)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        
        if method == 'grid':
            search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                xgb_model, param_grid, n_iter=n_iter, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1,
                random_state=42, verbose=1
            )
        
        print("Searching for best parameters...")
        search.fit(self.X_train, self.y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        # Evaluate
        y_pred = best_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"\nBest Parameters: {best_params}")
        print(f"Best CV Score (RMSE): ${np.sqrt(-search.best_score_):,.2f}")
        print(f"Test RMSE: ${rmse:,.2f}")
        print(f"Test R²: {r2:.4f}")
        
        self.best_models['XGBoost'] = best_model
        self.best_params['XGBoost'] = best_params
        
        return best_model, best_params
    
    def save_best_models(self, directory='tuned_models'):
        """Save tuned models"""
        os.makedirs(directory, exist_ok=True)
        
        print(f"\nSaving tuned models to '{directory}' directory...")
        for name, model in self.best_models.items():
            filename = os.path.join(directory, f"{name.lower().replace(' ', '_')}_tuned.pkl")
            joblib.dump(model, filename)
            print(f"  Saved: {filename}")
        
        # Save parameters
        import json
        params_file = os.path.join(directory, 'best_parameters.json')
        # Convert numpy types to Python types for JSON serialization
        params_dict = {}
        for model_name, params in self.best_params.items():
            params_dict[model_name] = {k: (int(v) if isinstance(v, (np.integer, np.int64)) 
                                          else (float(v) if isinstance(v, (np.floating, np.float64)) else v))
                                      for k, v in params.items()}
        
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=2)
        print(f"  Saved: {params_file}")
    
    def print_summary(self):
        """Print summary of tuning results"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*60)
        
        for name, params in self.best_params.items():
            print(f"\n{name}:")
            for param, value in params.items():
                print(f"  {param}: {value}")


def main():
    """Main execution"""
    print("="*60)
    print("Hyperparameter Tuning for Housing Price Prediction")
    print("RSK World - Free Programming Resources & Source Code")
    print("="*60)
    
    tuner = HyperparameterTuner()
    tuner.prepare_data()
    
    # Tune models (using randomized search for faster results)
    tuner.tune_random_forest(method='randomized', n_iter=30)
    tuner.tune_gradient_boosting(method='randomized', n_iter=30)
    
    if XGBOOST_AVAILABLE:
        tuner.tune_xgboost(method='randomized', n_iter=30)
    
    # Print summary
    tuner.print_summary()
    
    # Save models
    tuner.save_best_models()
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

