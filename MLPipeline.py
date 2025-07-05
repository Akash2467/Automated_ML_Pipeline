import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import joblib

# --- Load data safely ---
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except:
                file.seek(0)
                df = pd.read_csv(file, delimiter=';')
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

# --- Problem detection ---
def detect_problem(data, target):
    if target is None or target not in data.columns:
        return 'Clustering'
    target_series = data[target]
    if target_series.dtype == 'object' or target_series.nunique() <= 7:
        return 'Classification'
    else:
        return 'Regression'

# --- Preprocess ---
def preprocess_data(data, target):
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        raise ValueError("Dataset is empty after removing empty columns")
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    if numeric_cols:
        si = SimpleImputer(strategy='mean')
        data[numeric_cols] = si.fit_transform(data[numeric_cols])
    if cat_cols:
        si_c = SimpleImputer(strategy='most_frequent')
        data[cat_cols] = si_c.fit_transform(data[cat_cols])
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        if col == target:
            le_dict[col] = le
    return data, le_dict.get(target, None)

# --- Model training functions ---
def train_classification(model_name, xtrain, xtest, ytrain, ytest):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    model = models.get(model_name)
    model.fit(xtrain, ytrain)
    acc = accuracy_score(ytest, model.predict(xtest))
    return model, acc

def train_regression(model_name, xtrain, xtest, ytrain, ytest):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }
    model = models.get(model_name)
    model.fit(xtrain, ytrain)
    r2 = r2_score(ytest, model.predict(xtest))
    return model, r2

def train_clustering(model_name, data, n_clusters):
    if model_name == 'K Means':
        model = KMeans(n_clusters=n_clusters)
    elif model_name == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("Invalid model")
    preds = model.fit_predict(data)
    score = silhouette_score(data, preds) if len(set(preds)) > 1 else None
    return model, preds, score

# --- Save model ---
def save_model(model):
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/model.pkl")

# --- Start Streamlit App ---
st.title("AutoML Web App")

if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.problem_type = None
    st.session_state.model = None
    st.session_state.feature_names = []
    st.session_state.target_encoder = None

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state.df = df
        st.write("Preview:", df.head())
        columns = df.columns.tolist()
        target = st.selectbox("Select Target (None for Clustering)", ["None"] + columns)
        target = None if target == "None" else target
        df, target_encoder = preprocess_data(df, target)
        st.session_state.target_encoder = target_encoder
        st.session_state.problem_type = detect_problem(df, target)
        st.session_state.df = df

# After file is uploaded and preprocessed
if st.session_state.df is not None and st.session_state.problem_type:
    df = st.session_state.df
    problem_type = st.session_state.problem_type
    st.info(f"Detected Problem Type: {problem_type}")

    if problem_type in ["Classification", "Regression"]:
        X = df.drop(columns=[target])
        y = df[target]
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.feature_names = X.columns.tolist()

        if problem_type == "Classification":
            model_choice = st.selectbox("Select Classification Model", [
                'Logistic Regression', 'Decision Tree', 'Random Forest',
                'Support Vector Classifier', 'Gradient Boosting'
            ])
            if st.button("Train Classification Model"):
                model, acc = train_classification(model_choice, xtrain, xtest, ytrain, ytest)
                st.success(f"Trained {model_choice} with Accuracy: {acc:.2f}")
                save_model(model)
                st.session_state.model = model

        elif problem_type == "Regression":
            model_choice = st.selectbox("Select Regression Model", ['Linear Regression', 'Ridge', 'Lasso'])
            if st.button("Train Regression Model"):
                model, r2 = train_regression(model_choice, xtrain, xtest, ytrain, ytest)
                st.success(f"Trained {model_choice} with R² Score: {r2:.2f}")
                save_model(model)
                st.session_state.model = model

    elif problem_type == "Clustering":
        model_choice = st.selectbox("Select Clustering Model", ['K Means', 'Agglomerative'])
        n_clusters = st.number_input("Number of Clusters", min_value=2, value=3)
        if st.button("Train Clustering Model"):
            model, preds, score = train_clustering(model_choice, df, n_clusters)
            if score:
                st.success(f"Silhouette Score: {score:.2f}")
            else:
                st.warning("Could not compute Silhouette Score.")
            save_model(model)
            st.session_state.model = model

            # PCA scatter plot
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(df)
            plot_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
            plot_df['Cluster'] = preds
            fig, ax = plt.subplots()
            sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
            st.pyplot(fig)

