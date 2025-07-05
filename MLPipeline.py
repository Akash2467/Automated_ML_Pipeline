import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.metrics import accuracy_score,r2_score,silhouette_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import joblib
def load_data(file):
    if file.name.endswith('.csv') or file.name.endswith('.xls'):
        df=pd.read_csv(file)
    elif file.name.endswith('xlsx'):
        df=pd.read_excel(file)
    else:
        st.error("Unsupported File Type")
        return None
    return df
def detect_problem(data,target):
    try:
        if target is None or target not in data.columns:
            return 'Clustering'
        target_series=data[target]
        if target_series.dtype=='object':
            return 'Classification'
        unique=target_series.nunique()
        if unique<=7:
            return 'Classification'
        else:
            return 'Regression'
    except Exception as e:
        print('Error while detecting Problem :',e)
        return 'Unsupported'
    
def preprocess_data(data,target):
    try:
        data.dropna(axis=1,how='all',inplace=True)
        if data.empty:
            raise ValueError('Dataset is Empty after removing empty columns')
        numeric_columns=data.select_dtypes(include=['number']).columns.tolist()
        cat_columns=data.select_dtypes(include=['object']).columns.tolist()
        si=SimpleImputer(strategy='mean')
        si_c=SimpleImputer(strategy='most_frequent')
        if numeric_columns:
            data[numeric_columns]=si.fit_transform(data[numeric_columns])
        elif cat_columns:
            data[cat_columns]=si_c.fit_transform(data[cat_columns])
        le_dict={}
        for col in cat_columns:
            le=LabelEncoder()
            data[col]=le.fit_transform(data[col])
            if col==target:
                le_dict[col]=le
        return data,le_dict.get(target,None)
    except Exception as e:   
        print('Error While Pre-Processing : ',str(e))
        return None,None

def traintest(data,target):
    x=data.drop(target,axis=1)
    y=data[target]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
    return xtrain,xtest,ytrain,ytest



def train_classification(u_model,xtrain,xtest,ytrain,ytest):
    try:
        available_models={'Logistic Regression':LogisticRegression(),
                      'Decision Tree': DecisionTreeClassifier(),
                      'Random Forest':RandomForestClassifier(),
                      'Support Vector Classifier':SVC(),
                      'Gradient Boosting':GradientBoostingClassifier()}
        if u_model not in available_models:
            raise ValueError('Model Unavailable')
        else:
            model=available_models[u_model]
            model.fit(xtrain,ytrain)
            ypred=model.predict(xtest)
            accuracy=accuracy_score(ytest,ypred)
        return model,accuracy
    except Exception as e:
        print('Unable to Train Model.',str(e))
        return None,None
def train_regression(u_model,xtrain,xtest,ytrain,ytest):
    try:
        available_models={'Linear Regression':LinearRegression(),
                           'Ridge':Ridge(),
                           'Lasso':Lasso()}
        if u_model not in available_models:
            raise ValueError('Model Unavailable')
        else:
            model=available_models[u_model]
            model.fit(xtrain,ytrain)
            ypred=model.predict(xtest)
            r2score=r2_score(ytest,ypred)
        return model,r2score
    except Exception as e:
        print('Unable to Train Model.',str(e))
        return None,None
def train_clustering(u_model, data, n_clusters):
    try:
        if u_model == 'K Means':
            model = KMeans(n_clusters=n_clusters)
        elif u_model == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError('Model Unavailable')

        y_pred = model.fit_predict(data)
        unique_labels = set(y_pred)
        if len(unique_labels) > 1:
            silscore = silhouette_score(data, y_pred)
        else:
            silscore = None
            st.warning("Less than 2 clusters found. Silhouette score not available.")
        return model,y_pred,silscore
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None, None,None
def save_model(model,filename='trained_model.pkl',directory='saved_models'):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        path=os.path.join(directory,filename)
        joblib.dump(model,path)
        return path
    except Exception as e:
        print('Error saving the model : ',e)
        return None
def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)
    reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
    reduced_df['Cluster'] = labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='tab10')
    plt.title('Scatter Plot of Clusters (PCA-reduced)')
    st.pyplot(plt.gcf())

if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None

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

        df, target_encoder = preprocess_data(df, target)
        st.session_state.target_encoder = target_encoder

        problem_type = detect_problem(df, target)
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
                    st.success("Model Saved!")
                    st.session_state.model = model

            elif problem_type == "Regression":
                model_choice = st.selectbox("Select Regression Model", ['Linear Regression', 'Ridge', 'Lasso'])
                if st.button("Train Regression Model"):
                    model, r2 = train_regression(model_choice, xtrain, xtest, ytrain, ytest)
                    st.success(f"Trained {model_choice} with R² Score: {r2:.2f}")
                    save_model(model)
                    st.success("Model Saved!")
                    st.session_state.model = model

        elif problem_type == "Clustering":
            model_choice = st.selectbox("Select Clustering Model", ['K Means', 'Agglomerative'])
            n_clusters = st.number_input("Number of Clusters", min_value=2, value=3, step=1)
            if st.button("Train Clustering Model"):
                model, preds, score = train_clustering(model_choice, df, n_clusters)
                if score:
                    st.success(f"Silhouette Score: {score:.2f}")
                else:
                    st.warning("Could not compute Silhouette Score (only one cluster formed)")
                save_model(model)
                st.success("Model Saved!")
                st.session_state.model = model

                st.subheader("Cluster Scatter Plot")
                pca = PCA(n_components=2)
                data_reduced = pca.fit_transform(df)
                reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
                reduced_df['Cluster'] = preds
                fig, ax = plt.subplots()
                sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', ax=ax, palette='coolwarm')
                ax.set_title('Scatter Plot of Clusters')
                st.pyplot(fig)

if (
    "model" in st.session_state 
    and st.session_state.model is not None 
    and "feature_names" in st.session_state 
    and st.session_state.feature_names 
    and 'problem_type' in locals()
    and problem_type in ["Classification", "Regression"]
):
    st.header("Manual Input for Prediction (Using Trained Model)")
    user_input = {}
    st.subheader("Enter Feature Values")

    for feature in st.session_state.feature_names:
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
