# Smart Inventory Advisor for Retail Sales (Invense)

This repository contains a machine learning project that analyzes retail sales data to provide actionable inventory management recommendations. The primary goal is to help a store owner identify which products to restock and how many units to order, based on a data-driven sales forecast.

This project was developed for the **Datathon Nexus Crew**.

## Project Overview

The project follows a complete machine learning workflow:

1.  **Exploratory Data Analysis (EDA):** Initial analysis to understand trends, patterns, and relationships within the data.
2.  **Preprocessing & Feature Engineering:** Cleaning the data, converting categorical features to numerical values, and creating new time-based features from dates.
3.  **Model Training & Tuning:** Building a `RandomForestRegressor` to predict actual `Units Sold`. The model was tuned using `RandomizedSearchCV` to improve its reliability and prevent overfitting.
4.  **Actionable Recommendations:** Using the trained model to create a "Smart Inventory Advisor" that generates a prioritized list of products that need to be reordered.

## Model Description

The core of this project is a **Random Forest Regressor** (`scikit-learn`) trained to forecast the number of units sold (`Sales`) for a given product. It uses a variety of features to make its predictions, including:

* **Inventory & Order Data:** `Inventory Level`, `Units Ordered`
* **Pricing Factors:** `Price`, `Discount`, `Competitor Price`
* **Promotional & Environmental Factors:** `Holiday/Promotion`, `Weather`, `Seasonality`
* **Categorical Information:** `Product Category`, `Region`
* **Time-based Features:** `DayOfWeek`, `Month`, `Day`

## How It Works: The Smart Inventory Advisor

The final output is a "Store Owner's Action Plan." This tool automates the following process:

1.  **Analyzes All Products:** It iterates through every unique product in the dataset.
2.  **Forecasts Future Sales:** For each product, it predicts the expected sales for the next 30 days.
3.  **Calculates a Reorder Point:** It determines the stock level at which a reorder is necessary (based on a 7-day supply).
4.  **Generates an Action Plan:** It creates a simple table showing only the products that are below their reorder point and require immediate attention.

This provides a clear, data-driven to-do list that helps a business owner focus on the most critical inventory needs to maximize sales and prevent stockouts.

## How to Get Started with the Model

To use the pre-trained model, load the saved `.pkl` files.

```python
import pickle
import pandas as pd

# Load the trained model, scaler, and encoders
model = pickle.load(open('sales_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# --- Example: Prepare a single row of new data ---
# NOTE: This must have the same structure as the training data
new_data = pd.DataFrame([{
    'Inventory': 200, 'Orders': 50, 'Price': 35.0, 'Discount': 10, 
    'Competitor Price': 33.0, 'Promotion': 1, 'Category': 'Groceries', 
    'Region': 'North', 'Weather': 'Sunny', 'Seasonality': 'Spring',
    'DayOfWeek': 2, 'Month': 4, 'Day': 15
}])

# Apply the same preprocessing
for col in ['Category', 'Region', 'Weather', 'Seasonality']:
    new_data[col] = label_encoders[col].transform(new_data[col])

# Scale the features
new_data_scaled = scaler.transform(new_data)

# Make a prediction
predicted_sales = model.predict(new_data_scaled)
print(f"Predicted Sales: {predicted_sales[0]:.2f} units")
```

## Model Performance

The final model was tuned using `RandomizedSearchCV` to prevent overfitting. The key discovery was that sales data has a high degree of natural randomness, and the model successfully learned the predictable patterns.

| Metric         | Value   |
|----------------|---------|
| Training R²    | 0.6324  |
| **Testing R²** | **0.3402** |
| MSE            | 7811.96 |
| RMSE           | 88.39   |
| MAE            | 69.27   |

## Team (Nexus Crew)
- Hernicksen Satria
- Jeremy Wijaya
- Lawryan Andrew Darisang
