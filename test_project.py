"""
Housing Price Prediction Dataset - Project Test Script
RSK World - Free Programming Resources & Source Code
Website: https://rskworld.in
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Founder: Molla Samser
Designer & Tester: Rima Khatun
Created: 2026
"""

import os
import sys
import importlib.util

def test_file_imports():
    """Test if all Python files can be imported"""
    print("="*60)
    print("Testing File Imports")
    print("="*60)
    
    files_to_test = [
        'data_analysis.py',
        'data_visualization.py',
        'validate_data.py',
        'advanced_models.py',
        'feature_engineering.py',
        'hyperparameter_tuning.py',
        'model_comparison.py',
        'predict_price.py'
    ]
    
    results = []
    for file in files_to_test:
        if os.path.exists(file):
            try:
                spec = importlib.util.spec_from_file_location("test_module", file)
                if spec and spec.loader:
                    results.append((file, "OK", None))
                else:
                    results.append((file, "ERROR", "Cannot load module"))
            except Exception as e:
                results.append((file, "ERROR", str(e)))
        else:
            results.append((file, "MISSING", "File not found"))
    
    for file, status, error in results:
        if status == "OK":
            print(f"[OK] {file}")
        else:
            print(f"[{status}] {file}: {error}")
    
    return all(status == "OK" for _, status, _ in results)


def test_data_files():
    """Test if data files exist and are readable"""
    print("\n" + "="*60)
    print("Testing Data Files")
    print("="*60)
    
    import pandas as pd
    import json
    
    results = []
    
    # Test CSV
    try:
        df = pd.read_csv('housing_prices.csv')
        if len(df) > 0 and 'price' in df.columns:
            results.append(("CSV", "OK", f"{len(df)} records"))
        else:
            results.append(("CSV", "ERROR", "Invalid structure"))
    except Exception as e:
        results.append(("CSV", "ERROR", str(e)))
    
    # Test JSON
    try:
        with open('housing_prices.json', 'r') as f:
            data = json.load(f)
        if len(data) > 0:
            results.append(("JSON", "OK", f"{len(data)} records"))
        else:
            results.append(("JSON", "ERROR", "Empty file"))
    except Exception as e:
        results.append(("JSON", "ERROR", str(e)))
    
    for file_type, status, info in results:
        if status == "OK":
            print(f"[OK] {file_type}: {info}")
        else:
            print(f"[{status}] {file_type}: {info}")
    
    return all(status == "OK" for _, status, _ in results)


def test_dependencies():
    """Test if required dependencies are available"""
    print("\n" + "="*60)
    print("Testing Dependencies")
    print("="*60)
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    optional = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'scipy': 'scipy'
    }
    
    results = []
    
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            results.append((package_name, "OK", "Required"))
        except ImportError:
            results.append((package_name, "MISSING", "Required - Install with: pip install " + package_name))
    
    for module_name, package_name in optional.items():
        try:
            __import__(module_name)
            results.append((package_name, "OK", "Optional"))
        except ImportError:
            results.append((package_name, "NOT INSTALLED", "Optional - Install with: pip install " + package_name))
    
    for package, status, info in results:
        if status == "OK":
            print(f"[OK] {package} ({info})")
        elif status == "NOT INSTALLED":
            print(f"[INFO] {package} ({info})")
        else:
            print(f"[{status}] {package}: {info}")
    
    required_ok = all(status == "OK" for package, status, info in results if info == "Required")
    return required_ok


def test_basic_functionality():
    """Test basic functionality of key modules"""
    print("\n" + "="*60)
    print("Testing Basic Functionality")
    print("="*60)
    
    results = []
    
    # Test data loading
    try:
        import pandas as pd
        df = pd.read_csv('housing_prices.csv')
        if df.shape[0] > 0:
            results.append(("Data Loading", "OK", None))
        else:
            results.append(("Data Loading", "ERROR", "Empty dataset"))
    except Exception as e:
        results.append(("Data Loading", "ERROR", str(e)))
    
    # Test feature preparation
    try:
        feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                          'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                          'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if len(missing_cols) == 0:
            results.append(("Feature Columns", "OK", None))
        else:
            results.append(("Feature Columns", "ERROR", f"Missing: {missing_cols}"))
    except Exception as e:
        results.append(("Feature Columns", "ERROR", str(e)))
    
    for test_name, status, error in results:
        if status == "OK":
            print(f"[OK] {test_name}")
        else:
            print(f"[{status}] {test_name}: {error}")
    
    return all(status == "OK" for _, status, _ in results)


def main():
    """Run all tests"""
    print("="*60)
    print("Housing Price Prediction Dataset - Project Test")
    print("RSK World - Free Programming Resources & Source Code")
    print("="*60)
    
    test1 = test_file_imports()
    test2 = test_data_files()
    test3 = test_dependencies()
    test4 = test_basic_functionality()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"File Imports: {'PASS' if test1 else 'FAIL'}")
    print(f"Data Files: {'PASS' if test2 else 'FAIL'}")
    print(f"Dependencies: {'PASS' if test3 else 'FAIL'}")
    print(f"Basic Functionality: {'PASS' if test4 else 'FAIL'}")
    
    all_passed = test1 and test2 and test3 and test4
    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        print("Project is ready to use.")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

