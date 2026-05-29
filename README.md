# Used Car Price Prediction - End-to-End ML Pipeline

A complete machine learning project following the CRISP-DM process to predict used car prices for a USA-based dealership chain. The project covers business understanding, data preparation on 3 million listings, model comparison, and a deployed REST API with a user interface.

---

## Project Overview

Inconsistent pricing is a real business problem for used car dealerships. Pricing too high reduces turnover; pricing too low loses profit. This project builds a data-driven price prediction model that gives staff a consistent, feature-based estimate for any vehicle.

The final deployed model (tuned Random Forest on log-transformed price) achieves R2 = 0.9456 with a mean absolute error of approximately $2,442.

---

## Results Summary

| Model | R2 (Log Target) | MAE (USD) | RMSE (USD) |
|---|---|---|---|
| Random Forest (tuned) | 0.9456 | 2,441.82 | 5,823.26 |
| LightGBM (tuned) | 0.9421 | 2,655.48 | 6,095.34 |
| CatBoost (tuned) | 0.9350 | 2,850.10 | 6,492.39 |
| XGBoost (untuned) | 0.9313 | 2,968.26 | 6,725.43 |
| MLP Regressor | 0.8870 | 3,774.78 | 8,664.56 |
| Linear Regression | 0.5352 | 7,007.77 | 12,528.34 |

Log(price) target was used to stabilize training and reduce the effect of extreme values. All errors above are reported in USD after back-transformation.

---

## Dataset

- Source: US Used Cars dataset
- Size: 3,000,040 listings, 66 columns
- Final cleaned dataset: 2,743,730 rows, 17 selected features
- Key attributes: year, mileage, fuel type, body type, transmission, horsepower, engine displacement, accident/salvage history

---

## Pipeline Overview

### 1. Business Understanding
Defined the problem as a regression task with success criteria based on R2, MAE, and RMSE on unseen test data.

### 2. Data Understanding
- Scanned for duplicates, missing values, and format inconsistencies
- Key early findings: mileage negatively correlates with price; horsepower and year positively correlate with price

### 3. Data Preparation
- Loaded with Polars for performance on 3M rows
- Cleaned numeric fields stored as strings with units (e.g., "35.1 in", "18 gal")
- Handled missing values: mode for categorical columns, mean/median for numeric
- Removed outliers using IQR rule on mileage
- Engineered new features:
  - combined_fuel_economy
  - legroom
  - size_of_vehicle
  - major_options_count
  - has_incidents
- Reduced 66 columns to 17 meaningful, generalizable features

### 4. Modeling
Models tested: Linear Regression, Ridge, Random Forest, LightGBM, XGBoost, CatBoost, MLP Regressor, Keras neural network.

Training setup:
- 80/20 train/test split, same test set for all models
- Scaling applied for linear models and neural networks
- Grid search tuning with limited CV due to dataset size
- Both raw price and log(price) targets compared

### 5. Evaluation
- Tree-based and boosting models clearly outperformed linear models and neural networks on this tabular dataset
- Feature importance confirmed: horsepower, torque, year, and mileage are the strongest pricing signals
- Residual analysis showed well-centered errors for typical vehicles; spread increases for rare, high-value cars

### 6. Deployment
- Final model saved with joblib alongside feature column order and category mappings
- REST API built with FastAPI:
  - /health: uptime check
  - /codes: valid category values
  - /predict: returns predicted price in USD from JSON input
- API validated on 20,000 real rows; predictions match offline notebook results
- Most predictions fall within approximately 10 to 15 percent error

---

## Key Findings

- Year, mileage, horsepower, and torque dominate the pricing signal across all models
- Log(price) target significantly improves model stability and reduces the effect of luxury/rare vehicle outliers
- Tree-based models are strongly preferred over linear models and neural networks for this type of tabular regression task
- Hyperparameter tuning does not automatically improve results under limited compute conditions (tuned XGBoost performed worse than untuned in this setup)

---

## Project Structure

```
used-car-project/
|
|-- notebooks/
|   |-- 01_data_understanding.ipynb
|   |-- 02_data_preparation.ipynb
|   |-- 03_modeling.ipynb
|   |-- 04_evaluation.ipynb
|-- api/
|   |-- main.py              # FastAPI application
|   |-- model.joblib         # Saved Random Forest model
|   |-- feature_columns.json # Feature order used in training
|   |-- mappings.json        # Category encoding mappings
|-- img/
|-- Final_Report.md
|-- README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Uththara15/used-car-project.git
cd used-car-project

# Install dependencies
pip install pandas polars scikit-learn lightgbm xgboost catboost fastapi uvicorn joblib

# Run the API
uvicorn api.main:app --reload

# Send a prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2019, "mileage": 45000, "horsepower": 200, "fuel_type": "Gasoline", ...}'
```

---

## Tech Stack

- Python 3.x
- Polars (large dataset processing)
- Scikit-learn (Random Forest, MLP, Ridge, Linear Regression)
- LightGBM, XGBoost, CatBoost
- FastAPI (deployment)
- Pandas, NumPy
- Matplotlib, Seaborn

