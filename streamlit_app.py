import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

st.title("Diabetes Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# Show dataset
if st.checkbox("Show Raw Dataset"):
    st.subheader("Raw Data")
    st.write(df)

if st.checkbox(" Show Descriptive Statistics"):
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

# Sidebar - Graph Selection
st.sidebar.title(" Graph Controls")
show_hist = st.sidebar.checkbox("Histogram of All Features")
show_countplot = st.sidebar.checkbox("Countplot of Outcome")
show_boxplot = st.sidebar.checkbox("Boxplot: BMI vs Outcome")
show_heatmap = st.sidebar.checkbox("Correlation Heatmap")

# Histogram
if show_hist:
    st.subheader("Feature Distributions")
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.ravel()
    for i, col in enumerate(df.columns[:-1]):
        axs[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(col)
    st.pyplot(fig)

# Countplot
if show_countplot:
    st.subheader(" Outcome Countplot")
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Outcome", palette="Set2")
    st.pyplot(fig)

# Boxplot: BMI vs Outcome
if show_boxplot:
    st.subheader(" BMI vs Outcome Boxplot")
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Outcome", y="BMI", palette="Set3")
    st.pyplot(fig)

# Heatmap
if show_heatmap:
    st.subheader(" Correlation Heatmap")
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

data = load_data()

# Preprocessing
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.markdown(f"### Model Accuracy: {accuracy:.2f}")

# Prediction Section
st.header(" Predict Diabetes from User Input")

def user_input():
    pregnancies = st.slider("Pregnancies", 0, 17, 1)
    glucose = st.slider("Glucose", 0, 200, 120)
    bp = st.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 846, 79)
    bmi = st.slider("BMI", 0.0, 70.0, 30.1)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 10, 100, 33)

    features = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    scaled_features = scaler.transform(features)
    return scaled_features

input_data = user_input()

if st.button(" Predict"):
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.success(f"Prediction: **{result}**")

