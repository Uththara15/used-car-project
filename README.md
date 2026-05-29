# Used Car Price Prediction - End-to-End ML Pipeline

A complete machine learning project following the CRISP-DM process to predict used car prices for a USA-based dealership chain. Built as a group project at JAMK University of Applied Sciences. Personal contributions covered business understanding, modeling, deployment, and the final report.

The final deployed model (tuned Random Forest on log-transformed price) achieves R2 = 0.9456 with a mean absolute error of approximately $2,442.

---

## Personal Contributions

- Business Understanding: defined the problem, success criteria, and stakeholder goals
- Modeling: trained and compared 8 models including Random Forest, LightGBM, XGBoost, CatBoost, and neural networks; implemented log(price) target transformation
- Deployment: built and validated the FastAPI REST API and Streamlit UI; ran API validation on 20,000 real rows
- Final Report: wrote the complete project report covering all CRISP-DM phases

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

Log(price) target was used to stabilize training and reduce the effect of extreme values. All errors are reported in USD after back-transformation.

---

## Dataset

- Source: US Used Cars dataset
- Size: 3,000,040 listings, 66 columns
- Final cleaned dataset: 2,743,730 rows, 17 selected features
- Key attributes: year, mileage, fuel type, body type, transmission, horsepower, engine displacement, accident and salvage history

---

## Project Structure

```
used-car-project/
|
|-- project/
|   |-- 1 - Business Understanding/
|   |   |-- Report.md
|   |-- 2 - Data Understanding/
|   |   |-- Data Understanding.ipynb
|   |   |-- Report.md
|   |-- 3 - Data Preparation/
|   |   |-- Data Preparation.ipynb
|   |   |-- Data Preparation.md
|   |   |-- label_mappings.json
|   |-- 4 - Modeling/
|   |   |-- Modeling.ipynb
|   |   |-- Modeling_log.ipynb
|   |   |-- Summery.md
|   |-- 5 - Evaluation/
|   |   |-- Evaluation.ipynb
|   |   |-- Evaluation.md
|   |-- 6 - Deployment/
|   |   |-- deployment/
|   |   |   |-- api.py
|   |   |   |-- ui.py
|   |   |   |-- feature_columns.json
|   |   |   |-- mappings.json
|   |   |   |-- ui_defaults.json
|   |   |   |-- requirements.txt
|   |   |-- Deployment_validation.ipynb
|   |   |-- Report.md
|   |-- 7 - Final report/
|       |-- Final_report.md
|       |-- aida-cars-project.pptx
|-- img/
|-- README.md
```

---

## Pipeline Overview

### Business Understanding
Defined the core problem: inconsistent manual pricing at a used car dealership chain leads to lost profit or slow inventory turnover. Success criteria were set around R2, MAE, and RMSE on held-out test data.

### Data Understanding
Explored 3 million listings across 66 columns. Key early findings: mileage negatively correlates with price; horsepower and year positively correlate with price.

### Data Preparation
- Loaded with Polars for performance on 3M rows
- Cleaned numeric fields stored as strings with units (e.g. "35.1 in", "18 gal")
- Handled missing values: mode for categorical, mean/median for numeric
- Removed mileage outliers using the IQR rule
- Engineered new features: combined_fuel_economy, legroom, size_of_vehicle, major_options_count, has_incidents
- Reduced 66 columns to 17 meaningful features

### Modeling
Models compared: Linear Regression, Ridge, Random Forest, LightGBM, XGBoost, CatBoost, MLP Regressor, Keras neural network.

- 80/20 train/test split, same test set for all models
- Both raw price and log(price) targets compared
- Tree-based and boosting models clearly outperformed linear and neural network approaches

### Evaluation
- Tuned Random Forest on log(price) selected as the best model
- Feature importance confirmed: horsepower, torque, year, and mileage are the strongest pricing signals
- Residual analysis showed well-centered errors for typical vehicles; spread increases for rare, high-value cars

### Deployment
- Model saved with joblib alongside feature column order and category mappings
- REST API built with FastAPI:
  - /health: uptime check
  - /codes: valid category values
  - /predict: returns predicted price in USD from JSON input
- Streamlit UI for non-technical users
- API validated on 20,000 real rows; most predictions fall within 10 to 15 percent error

---

## Key Findings

- Year, mileage, horsepower, and torque dominate the pricing signal across all models
- Log(price) target significantly improves model stability and reduces the effect of outliers
- Tree-based models are strongly preferred over linear models and neural networks for tabular regression
- Hyperparameter tuning does not automatically improve results under limited compute conditions

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Uththara15/used-car-project.git
cd used-car-project

# Install dependencies
pip install -r project/6\ -\ Deployment/deployment/requirements.txt

# Run the API
cd project/6\ -\ Deployment/deployment
uvicorn api:app --reload

# Run the Streamlit UI
streamlit run ui.py
```

---

## Tech Stack

- Python 3.x
- Polars (large dataset processing)
- Scikit-learn (Random Forest, MLP, Ridge, Linear Regression)
- LightGBM, XGBoost, CatBoost
- FastAPI (REST API)
- Streamlit (UI)
- Pandas, NumPy
- Matplotlib, Seaborn
