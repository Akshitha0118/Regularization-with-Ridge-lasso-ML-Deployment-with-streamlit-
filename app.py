import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import statsmodels.api as sm

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Car MPG Analysis", layout="wide")

# ---------- BACKGROUND ----------
def set_bg(image):
    with open(image, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded}");
            background-size: cover;
        }}
        .block {{
            background: rgba(0,0,0,0.65);
            padding: 25px;
            border-radius: 15px;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("ml.webp")

# ---------- TITLE ----------
st.markdown(
    """
    <div class="block">
    <h1 style="text-align:center;">🚗 Car MPG Regression Analysis</h1>
    <p style="text-align:center;">
    Linear, Ridge & Lasso Regression with OLS Statistical Summary
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- LOAD & PREPROCESS DATA ----------
data = pd.read_csv("car-mpg.csv")
data = data.drop(['car_name'], axis=1)
data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
data = pd.get_dummies(data, columns=['origin'], dtype=int)
data = data.replace('?', np.nan)
data = data.apply(pd.to_numeric)
data = data.fillna(data.median())

X = data.drop('mpg', axis=1)
y = data['mpg']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)

# ---------- MODELS ----------
lin = LinearRegression().fit(x_train, y_train)
ridge = Ridge(alpha=0.3).fit(x_train, y_train)
lasso = Lasso(alpha=0.1).fit(x_train, y_train)

# ---------- COEFFICIENTS ----------
st.markdown("<div class='block'><h3>📌 Linear Regression Coefficients</h3></div>", unsafe_allow_html=True)
coef_df = pd.DataFrame({
    "Feature": x_train.columns,
    "Coefficient": lin.coef_
})
st.dataframe(coef_df)

# ---------- MODEL SCORES ----------
st.markdown("<div class='block'><h3>📊 Model Scores (R²)</h3></div>", unsafe_allow_html=True)

scores = pd.DataFrame({
    "Model": ["Linear", "Ridge", "Lasso"],
    "Train Score": [
        lin.score(x_train, y_train),
        ridge.score(x_train, y_train),
        lasso.score(x_train, y_train)
    ],
    "Test Score": [
        lin.score(x_test, y_test),
        ridge.score(x_test, y_test),
        lasso.score(x_test, y_test)
    ]
})
st.dataframe(scores)

# ---------- OLS SUMMARY ----------
st.markdown("<div class='block'><h3>📈 OLS Regression Summary</h3></div>", unsafe_allow_html=True)

X_ols = sm.add_constant(X.astype(float))
y_ols = y.astype(float)
ols_model = sm.OLS(y_ols, X_ols).fit()

st.text(ols_model.summary())

# ---------- RESIDUAL PLOTS ----------
st.markdown("<div class='block'><h3>📉 Residual Analysis</h3></div>", unsafe_allow_html=True)

y_pred = lin.predict(x_test)
residuals = y_test - y_pred

fig1, ax1 = plt.subplots()
sns.residplot(x=x_test['hp'], y=residuals, lowess=True, ax=ax1)
ax1.set_title("Residuals vs Horsepower")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.residplot(x=x_test['acc'], y=residuals, lowess=True, ax=ax2)
ax2.set_title("Residuals vs Acceleration")
st.pyplot(fig2)

# ---------- ACTUAL VS PREDICTED ----------
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred)
ax3.set_xlabel("Actual MPG")
ax3.set_ylabel("Predicted MPG")
ax3.set_title("Actual vs Predicted MPG")
st.pyplot(fig3)
