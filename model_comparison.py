"""
Housing Price Prediction Dataset - Model Comparison and Visualization
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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Try importing advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ModelComparator:
    """Compare multiple ML models with visualization"""
    
    def __init__(self, data_path='housing_prices.csv'):
        """Initialize comparator"""
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """Prepare data"""
        feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                          'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                          'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
        
        current_year = 2026
        X = self.df[feature_columns].copy()
        X['house_age'] = current_year - X['yr_built']
        X['total_sqft'] = X['sqft_above'] + X['sqft_basement']
        
        y = self.df['price']
        
        self.X = X
        self.y = y
        return X, y
    
    def initialize_models(self):
        """Initialize models to compare"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    
    def compare_models(self, cv_folds=5):
        """Compare models using cross-validation"""
        print("Comparing models with cross-validation...")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        comparison_data = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Cross-validation scores
            cv_rmse = -cross_val_score(model, self.X, self.y, cv=kfold, 
                                      scoring='neg_root_mean_squared_error')
            cv_r2 = cross_val_score(model, self.X, self.y, cv=kfold, scoring='r2')
            
            self.results[name] = {
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'cv_r2_mean': cv_r2.mean(),
                'cv_r2_std': cv_r2.std()
            }
            
            comparison_data.append({
                'Model': name,
                'CV RMSE Mean': f"${cv_rmse.mean():,.2f}",
                'CV RMSE Std': f"${cv_rmse.std():,.2f}",
                'CV R² Mean': f"{cv_r2.mean():.4f}",
                'CV R² Std': f"{cv_r2.std():.4f}"
            })
            
            print(f"  CV RMSE: ${cv_rmse.mean():,.2f} (±${cv_rmse.std():,.2f})")
            print(f"  CV R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def plot_comparison(self, save_dir='plots'):
        """Create comparison plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        rmse_means = [self.results[m]['cv_rmse_mean'] for m in model_names]
        rmse_stds = [self.results[m]['cv_rmse_std'] for m in model_names]
        r2_means = [self.results[m]['cv_r2_mean'] for m in model_names]
        r2_stds = [self.results[m]['cv_r2_std'] for m in model_names]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # RMSE Comparison
        axes[0, 0].barh(model_names, rmse_means, xerr=rmse_stds, capsize=5)
        axes[0, 0].set_xlabel('RMSE ($)')
        axes[0, 0].set_title('Model Comparison - RMSE (Lower is Better)')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # R² Comparison
        axes[0, 1].barh(model_names, r2_means, xerr=r2_stds, capsize=5, color='green')
        axes[0, 1].set_xlabel('R² Score')
        axes[0, 1].set_title('Model Comparison - R² Score (Higher is Better)')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # RMSE Bar Chart
        axes[1, 0].bar(model_names, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
        axes[1, 0].set_ylabel('RMSE ($)')
        axes[1, 0].set_title('RMSE by Model')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # R² Bar Chart
        axes[1, 1].bar(model_names, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='orange')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('R² Score by Model')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved: {filename}")
        plt.close()
        
        # Create ranking plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by RMSE
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['cv_rmse_mean'])
        sorted_names = [m[0] for m in sorted_models]
        sorted_rmse = [m[1]['cv_rmse_mean'] for m in sorted_models]
        sorted_rmse_std = [m[1]['cv_rmse_std'] for m in sorted_models]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_names)))
        bars = ax.barh(sorted_names, sorted_rmse, xerr=sorted_rmse_std, capsize=5, color=colors)
        
        ax.set_xlabel('RMSE ($)', fontsize=12)
        ax.set_title('Model Performance Ranking (Sorted by RMSE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_rmse)):
            ax.text(value + sorted_rmse_std[i] + 5000, bar.get_y() + bar.get_height()/2,
                   f'${value:,.0f}', va='center', fontsize=9)
        
        plt.tight_layout()
        filename = os.path.join(save_dir, 'model_ranking.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Ranking plot saved: {filename}")
        plt.close()


def main():
    """Main execution"""
    print("="*60)
    print("Model Comparison and Visualization")
    print("RSK World - Free Programming Resources & Source Code")
    print("="*60)
    
    comparator = ModelComparator()
    comparator.prepare_data()
    comparator.initialize_models()
    
    # Compare models
    comparison_df = comparator.compare_models(cv_folds=5)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Create visualizations
    comparator.plot_comparison()
    
    print("\n" + "="*60)
    print("Model Comparison Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

