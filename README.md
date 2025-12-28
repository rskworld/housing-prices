# Housing Price Prediction Dataset

**RSK World - Free Programming Resources & Source Code**  
**Website:** https://rskworld.in  
**Contact:** help@rskworld.in, support@rskworld.in  
**Phone:** +91 93305 39277  
**Founder:** Molla Samser  
**Designer & Tester:** Rima Khatun  
**Created:** 2026

## Overview

This dataset includes property features like size, bedrooms, bathrooms, location coordinates, neighborhood data, and sale prices. Perfect for regression models, feature engineering, and real estate analytics.

## Dataset Description

### Features

- **Property Features:**
  - `id`: Unique identifier for each property
  - `bedrooms`: Number of bedrooms
  - `bathrooms`: Number of bathrooms
  - `sqft_living`: Square footage of living area
  - `sqft_lot`: Square footage of lot
  - `floors`: Number of floors
  - `sqft_above`: Square footage above ground
  - `sqft_basement`: Square footage of basement

- **Location Data:**
  - `lat`: Latitude coordinate
  - `long`: Longitude coordinate
  - `zipcode`: ZIP code
  - `neighborhood`: Neighborhood name

- **Quality Features:**
  - `waterfront`: Whether property has waterfront view (0/1)
  - `view`: Quality of view (0-4)
  - `condition`: Overall condition (1-5)
  - `grade`: Overall grade (1-13)
  - `yr_built`: Year built
  - `yr_renovated`: Year renovated (0 if never renovated)

- **Comparative Features:**
  - `sqft_living15`: Average square footage of living area for 15 nearest neighbors
  - `sqft_lot15`: Average square footage of lot for 15 nearest neighbors

- **Target Variable:**
  - `price`: Sale price of the property

## File Structure

```
housing-prices/
├── housing_prices.csv          # Main dataset in CSV format
├── housing_prices.json         # Dataset in JSON format
├── data_analysis.py            # Python script for data analysis
├── data_visualization.py       # Python script for data visualization
├── housing_price_prediction.ipynb  # Jupyter notebook for interactive analysis
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── plots/                      # Generated visualization plots (created when running scripts)
```

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Python Scripts

**Basic Analysis:**
```bash
python data_analysis.py              # Basic data analysis and linear regression
python data_visualization.py         # Create data visualizations
python validate_data.py              # Validate dataset integrity
```

**Advanced Features (NEW in 2026):**
```bash
python advanced_models.py            # Advanced ML models (XGBoost, LightGBM, etc.)
python feature_engineering.py        # Advanced feature engineering
python hyperparameter_tuning.py      # Hyperparameter tuning with Grid/Randomized Search
python model_comparison.py           # Compare multiple models with visualization
```

### Jupyter Notebook

Open and run `housing_price_prediction.ipynb` for interactive analysis:
```bash
jupyter notebook housing_price_prediction.ipynb
```

## Technologies Used

- **CSV/JSON**: Data storage formats
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning framework (Linear Regression, Random Forest, Gradient Boosting, etc.)
- **XGBoost**: Advanced gradient boosting library
- **LightGBM**: Fast gradient boosting framework
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Jupyter Notebook**: Interactive analysis
- **Joblib**: Model persistence
- **SciPy**: Scientific computing utilities

## Model Performance

The dataset is suitable for:
- **Basic Models**: Linear Regression, Ridge, Lasso, Elastic Net
- **Tree-based Models**: Decision Tree, Random Forest, Gradient Boosting
- **Advanced Models**: XGBoost, LightGBM (optional)
- **Feature Engineering**: Create derived features, handle outliers, scaling, PCA
- **Hyperparameter Tuning**: Grid Search, Randomized Search
- **Model Comparison**: Cross-validation, performance metrics comparison
- **Real Estate Analytics**: Price prediction, feature importance analysis

## Dataset Statistics

- **Total Records:** 50 properties
- **Features:** 20 columns
- **Price Range:** $180,000 - $1,225,000
- **Average Price:** ~$460,000

## Use Cases

1. **Price Prediction:** Build regression models to predict house prices
2. **Feature Engineering:** Explore and create new features
3. **Data Analysis:** Analyze relationships between features and prices
4. **Visualization:** Create plots and charts for data insights
5. **Educational:** Learn machine learning and data science concepts

## License

This dataset is provided by RSK World for educational and research purposes.

## Contact

For questions, support, or feedback:
- **Email:** help@rskworld.in, support@rskworld.in
- **Phone:** +91 93305 39277
- **Website:** https://rskworld.in

---

**Created by RSK World**  
*Free Programming Resources & Source Code*

