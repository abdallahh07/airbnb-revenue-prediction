# Airbnb Revenue Prediction 🏠

## Overview
A machine learning project that predicts annual Airbnb listing revenue based on listing characteristics such as location, property size, ratings, and amenities. The dataset contains 90,000+ European Airbnb listings.

## Dataset
Download from Kaggle: https://www.kaggle.com/datasets/jasonairroi/airbnb-market-data-europe

## Project Structure
1. Data Loading
2. EDA - Past Rates Analysis
3. EDA - Listing Analysis
4. Data Cleaning & Imputation
5. Feature Engineering
6. Train/Test Split
7. Preprocessing (OneHotEncoder + ColumnTransformer)
8. Model Training & Evaluation
9. Model Analysis

## EDA Findings
- **Top cities:** Edinburgh and Palma generate the highest revenue per listing
- **Top countries:** France and Spain dominate total revenue
- **Superhosts earn ~60% more** than regular hosts on average
- **Entire homes** generate significantly more revenue than private or shared rooms
- **Revenue is right-skewed** — most listings earn under $50,000 annually
- **Peak season:** July-August show highest occupancy and reserved days

## Data Leakage
During development, an initial R2 of 0.99 was achieved. After investigation, this was identified as target leakage — columns like ttm_avg_rate, ttm_occupancy, and ttm_reserved_days mathematically derive revenue and were removed. The honest R2 after removal is 0.597. This demonstrates understanding of data integrity in ML pipelines.

## Model Results

| Model | R2 | RMSE |
|-------|-----|------|
| **CatBoost** | **0.597** | **$14,424** |
| XGBoost | 0.587 | $14,616 |
| Ridge | 0.281 | $19,273 |
| LinearRegression | 0.199 | $20,343 |

## Why R2 is 0.59
Revenue is influenced by factors not captured in listing data such as photo quality, host response time, local events, and competition. A 0.59 R2 with clean features is honest and realistic for Airbnb revenue prediction.

## Feature Engineering
- **wifi** — extracted from amenities list
- **Kitchen** — extracted from amenities list
- **size_score** — bedrooms + baths + beds
- **amenity_count** — number of amenities per listing

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder)
- CatBoost, XGBoost
- Matplotlib, Seaborn

## How to Run
```bash
pip install -r requirements.txt
python a.py
```

## What I Learned
- Identifying and handling target leakage in real datasets
- Building proper ML pipelines with ColumnTransformer
- Comparing multiple models systematically
- Feature engineering from raw text columns
