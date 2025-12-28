"""
Housing Price Prediction Dataset - Advanced Feature Engineering
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Advanced feature engineering for housing price prediction"""
    
    def __init__(self, data_path='housing_prices.csv'):
        """Initialize with dataset"""
        self.df = pd.read_csv(data_path)
        self.engineered_df = None
        self.scaler = None
        
    def create_basic_features(self):
        """Create basic engineered features"""
        print("Creating basic engineered features...")
        
        df = self.df.copy()
        current_year = 2026
        
        # Age features
        df['house_age'] = current_year - df['yr_built']
        df['years_since_renovation'] = df['yr_renovated'].apply(
            lambda x: current_year - x if x > 0 else df['house_age']
        )
        df['has_been_renovated'] = (df['yr_renovated'] > 0).astype(int)
        
        # Size features
        df['total_sqft'] = df['sqft_above'] + df['sqft_basement']
        df['sqft_ratio'] = df['sqft_above'] / (df['sqft_lot'] + 1)
        df['basement_ratio'] = df['sqft_basement'] / (df['total_sqft'] + 1)
        
        # Room features
        df['bedroom_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
        df['sqft_per_bedroom'] = df['sqft_living'] / (df['bedrooms'] + 0.1)
        df['sqft_per_bathroom'] = df['sqft_living'] / (df['bathrooms'] + 0.1)
        
        # Lot features
        df['lot_ratio'] = df['sqft_lot'] / (df['sqft_living'] + 1)
        df['living_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
        
        # Quality features
        df['quality_score'] = df['grade'] * df['condition']
        df['high_quality'] = (df['grade'] >= 8).astype(int)
        df['excellent_condition'] = (df['condition'] >= 4).astype(int)
        
        # Location features (normalized)
        df['lat_normalized'] = (df['lat'] - df['lat'].min()) / (df['lat'].max() - df['lat'].min())
        df['long_normalized'] = (df['long'] - df['long'].min()) / (df['long'].max() - df['long'].min())
        
        # Comparative features
        df['sqft_vs_neighbors'] = df['sqft_living'] / (df['sqft_living15'] + 1)
        df['lot_vs_neighbors'] = df['sqft_lot'] / (df['sqft_lot15'] + 1)
        df['larger_than_neighbors'] = (df['sqft_living'] > df['sqft_living15']).astype(int)
        
        # Interaction features
        df['sqft_grade_interaction'] = df['sqft_living'] * df['grade']
        df['view_waterfront'] = df['view'] * df['waterfront']
        
        # Price per square foot (for feature creation, not prediction)
        df['price_per_sqft'] = df['price'] / (df['sqft_living'] + 1)
        
        self.engineered_df = df
        print(f"Created {len(df.columns) - len(self.df.columns)} new features")
        return df
    
    def remove_outliers(self, columns=None, method='iqr'):
        """Remove outliers using IQR or Z-score method"""
        if self.engineered_df is None:
            self.create_basic_features()
        
        df = self.engineered_df.copy()
        initial_rows = len(df)
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col != 'id' and col != 'price']
        
        print(f"\nRemoving outliers using {method} method...")
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            try:
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[columns]))
                df = df[(z_scores < 3).all(axis=1)]
            except ImportError:
                print("  Warning: scipy not available. Install with: pip install scipy")
                print("  Falling back to IQR method...")
                return self.remove_outliers(columns, method='iqr')
        
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        self.engineered_df = df
        return df
    
    def scale_features(self, method='standard', columns=None):
        """Scale features using different methods"""
        if self.engineered_df is None:
            self.create_basic_features()
        
        df = self.engineered_df.copy()
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col not in ['id', 'price', 'zipcode']]
        
        print(f"\nScaling features using {method} method...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard', 'robust', or 'minmax'")
        
        df[columns] = self.scaler.fit_transform(df[columns])
        self.engineered_df = df
        print(f"Scaled {len(columns)} features")
        return df
    
    def apply_pca(self, n_components=0.95, columns=None):
        """Apply Principal Component Analysis"""
        if self.engineered_df is None:
            self.create_basic_features()
        
        df = self.engineered_df.copy()
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col not in ['id', 'price', 'zipcode']]
        
        print(f"\nApplying PCA...")
        X = df[columns].values
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create PCA column names
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        # Combine with non-numeric columns
        non_numeric_cols = [col for col in df.columns if col not in columns]
        result_df = pd.concat([df[non_numeric_cols], pca_df], axis=1)
        
        print(f"Reduced {len(columns)} features to {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        self.engineered_df = result_df
        self.pca = pca
        return result_df
    
    def get_feature_correlation(self, threshold=0.8):
        """Find highly correlated features"""
        if self.engineered_df is None:
            self.create_basic_features()
        
        numeric_cols = self.engineered_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.engineered_df[numeric_cols].corr().abs()
        
        # Find pairs with high correlation
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (>{threshold}):")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} <-> {col2}: {corr:.4f}")
        else:
            print(f"\nNo highly correlated pairs found (threshold: {threshold})")
        
        return high_corr_pairs
    
    def save_engineered_data(self, filename='housing_prices_engineered.csv'):
        """Save engineered dataset"""
        if self.engineered_df is None:
            self.create_basic_features()
        
        self.engineered_df.to_csv(filename, index=False)
        print(f"\nEngineered dataset saved to {filename}")
        return filename
    
    def get_feature_list(self):
        """Get list of all features"""
        if self.engineered_df is None:
            self.create_basic_features()
        
        return list(self.engineered_df.columns)


def main():
    """Main execution"""
    print("="*60)
    print("Advanced Feature Engineering")
    print("RSK World - Free Programming Resources & Source Code")
    print("="*60)
    
    engineer = FeatureEngineer()
    
    # Create features
    df = engineer.create_basic_features()
    print(f"\nOriginal features: {len(engineer.df.columns)}")
    print(f"Total features after engineering: {len(df.columns)}")
    
    # Check correlations
    engineer.get_feature_correlation(threshold=0.8)
    
    # Save engineered data
    engineer.save_engineered_data()
    
    print("\n" + "="*60)
    print("Feature Engineering Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

