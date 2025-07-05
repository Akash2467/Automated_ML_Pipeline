# Automated_ML_Pipeline
This is a Streamlit-based web application that automates the machine learning workflow for classification, regression, and clustering tasks. Users can upload a dataset, select the target variable (or leave it blank for clustering), and the app will handle preprocessing, model training, evaluation, and predictions through an interactive interface.

## Features

- Upload `.csv`, `.xls`, or `.xlsx` datasets
- Automatically detects the problem type (Classification, Regression, or Clustering)
- Handles missing values (mean for numeric, mode for categorical)
- Label encodes categorical features
- Supports model training:
  - Classification: Logistic Regression, Decision Tree, Random Forest, SVC, Gradient Boosting
  - Regression: Linear Regression, Ridge, Lasso
  - Clustering: K-Means, Agglomerative Clustering
- Displays evaluation metrics and visualizations
- Saves trained models locally
- Allows manual input for predictions

## Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

## How to Run

Make sure you have Streamlit installed, then run:

```bash
streamlit run app.py
