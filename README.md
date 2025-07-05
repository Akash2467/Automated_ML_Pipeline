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
- Displays 2D scatter plot of clusters using PCA
## Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

## Preffered Dataset Format
Preferred Dataset Format
To ensure smooth execution and avoid errors during model training and prediction, it is recommended to upload datasets that:

Are in .csv, .xlsx, or .xls format.

Do not contain mixed delimiters (e.g., both commas and semicolons).

Have a single header row (avoid files with metadata or multiple headers).

Contain well-formatted tabular data with clear column names.

Avoid entirely empty columns or rows.

Do not include extremely large text fields or unstructured data in a single column.

## Live Demo
[Click here to try the app](https://automatedmlpipeline-qmde2kwomymjoq5xfaou4j.streamlit.app/)

## How to Run

Make sure you have Streamlit installed, then run:


```bash
streamlit run MLPipeline.py
