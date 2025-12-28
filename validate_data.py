"""
Housing Price Prediction Dataset - Data Validation Script
RSK World - Free Programming Resources & Source Code
Website: https://rskworld.in
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Founder: Molla Samser
Designer & Tester: Rima Khatun
Created: 2026
"""

import pandas as pd
import json
import os

def validate_csv():
    """Validate CSV file"""
    print("Validating CSV file...")
    try:
        df = pd.read_csv('housing_prices.csv')
        print(f"[OK] CSV file loaded successfully")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Missing values: {df.isnull().sum().sum()}")
        print(f"  - Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        return True
    except Exception as e:
        print(f"[ERROR] Error validating CSV: {e}")
        return False

def validate_json():
    """Validate JSON file"""
    print("\nValidating JSON file...")
    try:
        with open('housing_prices.json', 'r') as f:
            data = json.load(f)
        print(f"[OK] JSON file loaded successfully")
        print(f"  - Records: {len(data)}")
        print(f"  - First record keys: {list(data[0].keys())}")
        return True
    except Exception as e:
        print(f"[ERROR] Error validating JSON: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    print("\nChecking required files...")
    required_files = [
        'housing_prices.csv',
        'housing_prices.json',
        'data_analysis.py',
        'data_visualization.py',
        'housing_price_prediction.ipynb',
        'advanced_models.py',
        'feature_engineering.py',
        'hyperparameter_tuning.py',
        'model_comparison.py',
        'predict_price.py',
        'README.md',
        'requirements.txt',
        'index.html'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[MISSING] {file}")
            all_exist = False
    
    return all_exist

def main():
    print("="*60)
    print("Housing Price Prediction Dataset - Validation")
    print("RSK World - Free Programming Resources & Source Code")
    print("="*60)
    
    csv_ok = validate_csv()
    json_ok = validate_json()
    files_ok = check_required_files()
    
    print("\n" + "="*60)
    if csv_ok and json_ok and files_ok:
        print("[SUCCESS] All validations passed!")
        print("Dataset is ready to use.")
    else:
        print("[FAILED] Some validations failed. Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()

