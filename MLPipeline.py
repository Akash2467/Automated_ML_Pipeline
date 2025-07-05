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
from sklearn.decomposition import PCA
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Load data function
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

# Problem detection
def detect_problem(data, target):
    try:
        if target is None or target not in data.columns:
            return 'Clustering'
        target_series = data[target]
        if target_series.dtype == 'object':
            return 'Classification'
        unique = target_series.nunique()
        return 'Classification' if unique <= 7 else 'Regression'
    except Exception as e:
        print('Error while detecting Problem:', e)
        return 'Unsupported'

# Preprocessing
def preprocess_data(data, target):
    try:
        data.dropna(axis=1, how='all', inplace=True)
        if data.empty:
            raise ValueError("Dataset is empty after removing empty columns.")
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
            le_dict[col] = le

        target_encoder = le_dict.pop(target, None) if target in le_dict else None
        return data, target_encoder, le_dict
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None, None

# Classification model
def train_classification(u_model, xtrain, xtest, ytrain, ytest):
    try:
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Classifier': SVC(),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        model = models[u_model]
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        acc = accuracy_score(ytest, ypred)
        return model, acc
    except Exception as e:
        st.error(f"Classification training failed: {e}")
        return None, None

# Regression model
def train_regression(u_model, xtrain, xtest, ytrain, ytest):
    try:
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso()
        }
        model = models[u_model]
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        score = r2_score(ytest, ypred)
        return model, score
    except Exception as e:
        st.error(f"Regression training failed: {e}")
        return None, None

# Clustering
def train_clustering(u_model, data, n_clusters):
    try:
        if u_model == 'K Means':
            model = KMeans(n_clusters=n_clusters)
        elif u_model == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError("Model Unavailable")

        y_pred = model.fit_predict(data)
        score = silhouette_score(data, y_pred) if len(set(y_pred)) > 1 else None
        return model, y_pred, score
    except Exception as e:
        st.error(f"Clustering error: {e}")
        return None, None, None

# Save model
def save_model(model, filename='trained_model.pkl', directory='saved_models'):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, filename)
        joblib.dump(model, path)
        return path
    except Exception as e:
        st.error(f"Model saving failed: {e}")
        return None

# Plot clusters
def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    reduced_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    reduced_df['Cluster'] = labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='tab10')
    plt.title('Scatter Plot of Clusters (PCA-reduced)')
    st.pyplot(plt.gcf())

# Streamlit state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}

# UI
st.title("AutoML Web App")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        target = st.selectbox("Select Target Column (Leave empty for Clustering)", ["None"] + columns)
        target = None if target == "None" else target

        df, target_encoder, feature_encoders = preprocess_data(df, target)
        st.session_state.target_encoder = target_encoder
        st.session_state.feature_encoders = feature_encoders

        problem_type = detect_problem(df, target)
        st.info(f"Detected Problem Type: {problem_type}")

        if problem_type in ["Classification", "Regression"]:
            X = df.drop(columns=[target])
            y = df[target]
            xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state.feature_names = X.columns.tolist()

            if problem_type == "Classification":
                model_choice = st.selectbox("Select Classification Model", list(train_classification.__annotations__.keys())[1:])
                if st.button("Train Classification Model"):
                    model, acc = train_classification(model_choice, xtrain, xtest, ytrain, ytest)
                    if model and acc is not None:
                        st.success(f"Trained {model_choice} with Accuracy: {acc:.2f}")
                        save_model(model)
                        st.success("Model Saved!")
                        st.session_state.model = model
                    else:
                        st.error("Training failed.")

            elif problem_type == "Regression":
                model_choice = st.selectbox("Select Regression Model", ['Linear Regression', 'Ridge', 'Lasso'])
                if st.button("Train Regression Model"):
                    model, score = train_regression(model_choice, xtrain, xtest, ytrain, ytest)
                    if model and score is not None:
                        st.success(f"Trained {model_choice} with R² Score: {score:.2f}")
                        save_model(model)
                        st.success("Model Saved!")
                        st.session_state.model = model
                    else:
                        st.error("Training failed.")

        elif problem_type == "Clustering":
            model_choice = st.selectbox("Select Clustering Model", ['K Means', 'Agglomerative'])
            n_clusters = st.number_input("Number of Clusters", min_value=2, value=3, step=1)
            if st.button("Train Clustering Model"):
                model, preds, score = train_clustering(model_choice, df, n_clusters)
                if model and preds is not None:
                    if score:
                        st.success(f"Silhouette Score: {score:.2f}")
                    else:
                        st.warning("Silhouette Score could not be calculated.")
                    save_model(model)
                    st.success("Model Saved!")
                    st.session_state.model = model

                    st.subheader("Cluster Scatter Plot")
                    plot_clusters(df, preds)
                else:
                    st.error("Clustering failed.")

# Manual prediction
if (
    st.session_state.model is not None 
    and st.session_state.feature_names 
    and problem_type in ["Classification", "Regression"]
):
    st.header("Manual Input for Prediction")
    user_input = {}
    for feature in st.session_state.feature_names:
        if feature in st.session_state.feature_encoders:
            encoder = st.session_state.feature_encoders[feature]
            category = st.selectbox(f"{feature}", encoder.classes_, key=f"manual_{feature}")
            user_input[feature] = encoder.transform([category])[0]
        else:
            user_input[feature] = st.number_input(f"{feature}", key=f"manual_{feature}")

    if st.button("Predict from Manual Input"):
        try:
            input_df = pd.DataFrame([user_input])
            prediction = st.session_state.model.predict(input_df)
            if problem_type == "Classification" and st.session_state.target_encoder:
                prediction = st.session_state.target_encoder.inverse_transform(prediction)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

