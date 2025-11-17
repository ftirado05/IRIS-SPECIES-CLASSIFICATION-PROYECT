import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIG

st.markdown("---")
st.markdown("**Developed by:** Fabian Tirado · Juan Rodriguez · Esteban Suarez")
st.markdown("---")
st.set_page_config(page_title="Iris Species Classification", layout="wide")

st.title("Iris Species Classification Proyect - Streamlit")
st.markdown("""
This app trains a Random Forest classifier on the Iris dataset, shows evaluation metrics,
and allows users to input measurements to predict the species and visualize the new sample
in a 3D scatter plot alongside the dataset.
""")

# LOAD DATA

@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']
    df['species'] = df['target'].map({i:name for i,name in enumerate(iris.target_names)})
    return df, iris

df, iris = load_data()


# SIDEBAR
st.sidebar.header("Model & Input Settings")

test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state (seed)", value=42, min_value=0)
n_estimators = st.sidebar.slider("RF n_estimators", 10, 300, 100, 10)
max_depth = st.sidebar.slider("RF max_depth (0 = None)", 0, 30, 0, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Enter measurements for prediction")

input_sepal_length = st.sidebar.number_input("Sepal length (cm)", min_value=0.0, value=5.1)
input_sepal_width  = st.sidebar.number_input("Sepal width (cm)",  min_value=0.0, value=3.5)
input_petal_length = st.sidebar.number_input("Petal length (cm)", min_value=0.0, value=1.4)
input_petal_width  = st.sidebar.number_input("Petal width (cm)",  min_value=0.0, value=0.2)

user_sample = np.array([[input_sepal_length, input_sepal_width, input_petal_length, input_petal_width]])

# PREPROCESSING
X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
user_sample_scaled = scaler.transform(user_sample)

# MODEL TRAINING
rf = RandomForestClassifier(
    n_estimators=int(n_estimators),
    max_depth=None if max_depth == 0 else int(max_depth),
    random_state=int(random_state)
)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

# METRICS
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

col1, col2 = st.columns((1, 1))

with col1:
    st.subheader("Model evaluation")
    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("Precision", f"{prec:.3f}")
    st.metric("Recall", f"{rec:.3f}")
    st.metric("F1-score", f"{f1:.3f}")

    st.markdown("### Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm, 
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=iris.target_names,
        y=iris.target_names
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.markdown("### Scatter Matrix")
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=['sepal_length','sepal_width','petal_length','petal_width'],
        color='species'
    )
    fig_matrix.update_traces(diagonal_visible=False)
    st.plotly_chart(fig_matrix, use_container_width=True)

# USER PREDICTION
pred_class = rf.predict(user_sample_scaled)[0]
pred_proba = rf.predict_proba(user_sample_scaled)[0]
pred_species = iris.target_names[pred_class]

st.subheader("Prediction for your input")
st.write(f"**Predicted species:** {pred_species}")

proba_df = pd.DataFrame({
    "species": iris.target_names,
    "probability": pred_proba
})
st.dataframe(proba_df)

# 3D SCATTER PLOT
plotdf = df.copy()
plotdf["sample"] = "dataset"

user_row = {
    "sepal_length": input_sepal_length,
    "sepal_width": input_sepal_width,
    "petal_length": input_petal_length,
    "petal_width": input_petal_width,
    "species": pred_species,
    "sample": "user"
}

plotdf = pd.concat([plotdf, pd.DataFrame([user_row])], ignore_index=True)

fig3d = px.scatter_3d(
    plotdf,
    x="petal_length",
    y="petal_width",
    z="sepal_length",
    color="species",
    symbol="sample",
    size=[10 if s == "user" else 5 for s in plotdf["sample"]],
)
fig3d.update_layout(height=600)

st.subheader("3D Visualization of Your Sample")
st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")
st.markdown("**Developed by:** Fabian Tirado · Juan Rodriguez · Esteban Suarez")
