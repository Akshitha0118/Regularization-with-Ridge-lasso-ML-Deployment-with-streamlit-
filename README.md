# Regularization-with-Ridge-lasso-ML-Deployment-with-streamlit-
Day - 3  Regularization with ( Ridge &amp; lasso ) Machine Learning project with Car-Mpg  dataset predictions Deployment with Streamlit

# streamlit live link:https://re7nymugpuy4yaszdbewj7.streamlit.app/

---

# 🚗 Car MPG Regression Analysis using Streamlit

A complete **Machine Learning regression analysis dashboard** built with **Streamlit**, showcasing **Linear, Ridge, and Lasso Regression** along with a detailed **OLS statistical summary** using `statsmodels`.

This project focuses on understanding the key factors affecting **car fuel efficiency (MPG)** through both **predictive modeling** and **statistical interpretation**.

---

## 📌 Features

- 📊 Interactive **Streamlit Web App**
- 🔍 **Exploratory Data Analysis**
- 📈 Regression Models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
- 📐 **OLS Regression Summary** (p-values, t-stats, R², Adjusted R²)
- 📉 Residual Diagnostics:
  - Residuals vs Horsepower
  - Residuals vs Acceleration
- 🔁 Actual vs Predicted MPG visualization
- 🎨 Custom background styling using CSS & Base64 images

---

## 🧠 Technologies & Libraries Used

- **Python**
- **Streamlit**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **Statsmodels**

---

## 📂 Dataset

- **Dataset Name:** Car MPG Dataset  
- **Target Variable:** `mpg`
- **Features Include:**
  - Cylinders
  - Displacement
  - Horsepower
  - Weight
  - Acceleration
  - Model Year
  - Origin (One-Hot Encoded)

Missing values are handled using **median imputation**, and categorical variables are converted using **one-hot encoding**.

---

## ⚙️ Model Workflow

1. Load and preprocess the dataset
2. Handle missing values and encode categorical features
3. Split data into training and testing sets
4. Train regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
5. Evaluate models using **R² score**
6. Perform **OLS regression analysis**
7. Visualize residuals and prediction accuracy

---

## 📈 OLS Regression Analysis

The project uses **Statsmodels OLS** to provide:
- Coefficient estimates
- Standard errors
- t-statistics
- p-values
- Confidence intervals

This helps in **statistical interpretation** and **feature significance analysis**, beyond just prediction accuracy.
---
## 🚀 Key Insights

- Weight and horsepower have a strong negative impact on MPG

- Newer model years show improved fuel efficiency

- OLS analysis helps validate statistically significant predictors

- Ridge and Lasso help control multicollinearity


