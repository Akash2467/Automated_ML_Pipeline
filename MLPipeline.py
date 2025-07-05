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

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'df' not in st.session_state:
    st.session_state.df = None

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            try:
                return pd.read_csv(file)
            except pd.errors.ParserError:
                file.seek(0)
                return pd.read_csv(file, delimiter=';')
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        else:
            st.error("Unsupported file type.")
            return None
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

def detect_problem(data, target):
    if target is None or target not in data.columns:
        return 'Clustering'
    target_series = data[target]
    if target_series.dtype == 'object':
        return 'Classification'
    unique = target_series.nunique()
    return 'Classification' if unique <= 7 else 'Regression'

def preprocess_data(data, target):
    try:
        data.dropna(axis=1, how='all', inplace=True)
        if data.empty:
            raise ValueError('Dataset is empty after removing empty columns')
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        cat_columns = data.select_dtypes(include=['object']).columns.tolist()
        si = SimpleImputer(strategy='mean')
        si_c = SimpleImputer(strategy='most_frequent')
        if numeric_columns:
            data[numeric_columns] = si.fit_transform(data[numeric_columns])
        if cat_columns:
            data[cat_columns] = si_c.fit_transform(data[cat_columns])
        le_dict = {}
        for col in cat_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            if col == target:
                le_dict[col] = le
        return data, le_dict.get(target, None)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None

def train_classification(model_name, xtrain, xtest, ytrain, ytest):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    model = models.get(model_name)
    if model:
        model.fit(xtrain, ytrain)
        acc = accuracy_score(ytest, model.predict(xtest))
        return model, acc
    return None, None

def train_regression(model_name, xtrain, xtest, ytrain, ytest):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }
    model = models.get(model_name)
    if model:
        model.fit(xtrain, ytrain)
        r2 = r2_score(ytest, model.predict(xtest))
        return model, r2
    return None, None

def train_clustering(model_name, data, n_clusters):
    try:
        if model_name == 'K Means':
            model = KMeans(n_clusters=n_clusters)
        elif model_name == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            return None, None, None
        y_pred = model.fit_predict(data)
        score = silhouette_score(data, y_pred) if len(set(y_pred)) > 1 else None
        return model, y_pred, score
    except Exception as e:
        st.error(f"Clustering error: {e}")
        return None, None, None

def save_model(model):
    path = 'saved_models/trained_model.pkl'
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(model, path)
    return path

def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    df['Cluster'] = labels
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    st.pyplot(fig)

st.title("AutoML Web App")

file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if file:
    df = load_data(file)
    if df is not None:
        st.session_state.df = df
        st.dataframe(df.head())
        target = st.selectbox("Select Target Column (Leave empty for Clustering)", ["None"] + df.columns.tolist())
        target = None if target == "None" else target

        df_cleaned, target_encoder = preprocess_data(df.copy(), target)
        if df_cleaned is not None:
            st.session_state.target_encoder = target_encoder
            st.session_state.problem_type = detect_problem(df_cleaned, target)
            st.info(f"Detected Problem Type: {st.session_state.problem_type}")

            if st.session_state.problem_type in ["Classification", "Regression"]:
                X = df_cleaned.drop(columns=[target])
                y = df_cleaned[target]
                st.session_state.feature_names = X.columns.tolist()
                xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

                if st.session_state.problem_type == "Classification":
                    model_choice = st.selectbox("Select Classification Model", list({
                        'Logistic Regression', 'Decision Tree', 'Random Forest', 
                        'Support Vector Classifier', 'Gradient Boosting'}))
                    if st.button("Train Classification Model"):
                        model, acc = train_classification(model_choice, xtrain, xtest, ytrain, ytest)
                        if acc is not None:
                            st.success(f"Trained {model_choice} with Accuracy: {acc:.2f}")
                            save_model(model)
                            st.session_state.model = model

                elif st.session_state.problem_type == "Regression":
                    model_choice = st.selectbox("Select Regression Model", ['Linear Regression', 'Ridge', 'Lasso'])
                    if st.button("Train Regression Model"):
                        model, r2 = train_regression(model_choice, xtrain, xtest, ytrain, ytest)
                        if r2 is not None:
                            st.success(f"Trained {model_choice} with R² Score: {r2:.2f}")
                            save_model(model)
                            st.session_state.model = model

            elif st.session_state.problem_type == "Clustering":
                model_choice = st.selectbox("Select Clustering Model", ['K Means', 'Agglomerative'])
                n_clusters = st.number_input("Number of Clusters", min_value=2, value=3, step=1)
                if st.button("Train Clustering Model"):
                    model, labels, score = train_clustering(model_choice, df_cleaned, n_clusters)
                    if score:
                        st.success(f"Silhouette Score: {score:.2f}")
                    save_model(model)
                    st.session_state.model = model
                    st.subheader("Cluster Visualization")
                    plot_clusters(df_cleaned, labels)
