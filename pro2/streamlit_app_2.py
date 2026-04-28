import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Bank Retention Dashboard", layout="wide")

st.title("💳 Customer Retention Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

df = load_data()

# -------------------------------
# DATA CLEANING
# -------------------------------
df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df['EngagementScore'] = (
    df['IsActiveMember'] +
    df['HasCrCard'] +
    df['NumOfProducts']
)

# -------------------------------
# MODEL TRAINING
# -------------------------------
features = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain',
    'Gender_Male',
    'EngagementScore'
]

X = df[features]
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🎛️ Filters")

product_filter = st.sidebar.slider("Products", 1, 4, (1, 4))

balance_filter = st.sidebar.slider(
    "Balance",
    int(df['Balance'].min()),
    int(df['Balance'].max()),
    (0, int(df['Balance'].max()))
)

activity_filter = st.sidebar.selectbox("Active Member", ["All", 1, 0])

geo_filter = st.sidebar.multiselect(
    "Geography",
    ["France", "Germany", "Spain"],
    default=["France", "Germany", "Spain"]
)

# -------------------------------
# APPLY FILTERS (FIXED)
# -------------------------------
filtered_df = df.copy()

# Product filter
filtered_df = filtered_df[
    (filtered_df['NumOfProducts'] >= product_filter[0]) &
    (filtered_df['NumOfProducts'] <= product_filter[1])
]

# Balance filter
filtered_df = filtered_df[
    (filtered_df['Balance'] >= balance_filter[0]) &
    (filtered_df['Balance'] <= balance_filter[1])
]

# Activity filter
if activity_filter != "All":
    filtered_df = filtered_df[
        filtered_df['IsActiveMember'] == activity_filter
    ]

# Geography filter (correct handling)
geo_conditions = []

if "France" in geo_filter:
    geo_conditions.append(
        (filtered_df['Geography_Germany'] == 0) &
        (filtered_df['Geography_Spain'] == 0)
    )

if "Germany" in geo_filter:
    geo_conditions.append(filtered_df['Geography_Germany'] == 1)

if "Spain" in geo_filter:
    geo_conditions.append(filtered_df['Geography_Spain'] == 1)

if geo_conditions:
    filtered_df = filtered_df[np.logical_or.reduce(geo_conditions)]

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

active_churn = filtered_df[filtered_df['IsActiveMember']==1]['Exited'].mean()
inactive_churn = filtered_df[filtered_df['IsActiveMember']==0]['Exited'].mean()

col1.metric("Total Customers", len(filtered_df))
col2.metric("Churn Rate", round(filtered_df['Exited'].mean(),2))
col3.metric("Engagement Gap", round(inactive_churn - active_churn,2))
col4.metric("Model Accuracy", round(accuracy,2))

# -------------------------------
# CHARTS
# -------------------------------
col5, col6 = st.columns(2)

with col5:
    fig1 = px.bar(
        filtered_df.groupby('IsActiveMember')['Exited'].mean().reset_index(),
        x='IsActiveMember',
        y='Exited',
        title="Churn by Activity",
        color='Exited'
    )
    st.plotly_chart(fig1, use_container_width=True)

with col6:
    fig2 = px.line(
        filtered_df.groupby('NumOfProducts')['Exited'].mean().reset_index(),
        x='NumOfProducts',
        y='Exited',
        markers=True,
        title="Churn by Products"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# HIGH VALUE RISK
# -------------------------------
st.subheader("⚠️ High Value Risk Customers")

risk_df = filtered_df[
    (filtered_df['Balance'] > 100000) &
    (filtered_df['IsActiveMember'] == 0)
]

st.write(f"High Risk Customers: {len(risk_df)}")
st.dataframe(risk_df.head(10))

# -------------------------------
# DATA TABLE
# -------------------------------
st.subheader("📋 Filtered Data")
st.dataframe(filtered_df.head(50))

# -------------------------------
# PREDICTION
# -------------------------------
st.subheader("🤖 Predict Customer Churn")

col7, col8 = st.columns(2)

with col7:
    credit = st.number_input("Credit Score", 300, 900, 600)
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure", 0, 10, 3)
    balance = st.number_input("Balance", 0, 200000, 50000)

with col8:
    products = st.slider("Products", 1, 4, 1)
    card = st.selectbox("Has Credit Card", [0,1])
    active = st.selectbox("Active Member", [0,1])
    salary = st.number_input("Salary", 10000, 200000, 50000)

geo = st.selectbox("Geography", ["France","Germany","Spain"])
gender = st.selectbox("Gender", ["Female","Male"])

geo_germany = 1 if geo=="Germany" else 0
geo_spain = 1 if geo=="Spain" else 0
gender_male = 1 if gender=="Male" else 0

input_df = pd.DataFrame({
    'CreditScore':[credit],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[products],
    'HasCrCard':[card],
    'IsActiveMember':[active],
    'EstimatedSalary':[salary],
    'Geography_Germany':[geo_germany],
    'Geography_Spain':[geo_spain],
    'Gender_Male':[gender_male],
    'EngagementScore':[active + card + products]
})

# IMPORTANT: match training columns
input_df = input_df.reindex(columns=features, fill_value=0)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    
    if pred == 1:
        st.error("❌ High Risk: Customer will CHURN")
    else:
        st.success("✅ Customer will STAY")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Customer Retention Analysis Project")