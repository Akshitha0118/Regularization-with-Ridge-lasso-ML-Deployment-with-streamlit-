import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score

# read the dataset 
data = pd.read_csv(r'C:\Users\ADMIN\Downloads\car-mpg.csv')

# drop attributes , fill missing values , convert numeric values 
data = data.drop(['car_name'],axis=1)
data['origin']=data['origin'].replace({1:'america', 2:'europe', 3:'asia'})
data = pd.get_dummies(data,columns=['origin'],dtype=int)
data = data.replace('?',np.nan)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.median())

data = data.apply(pd.to_numeric , errors = 'ignore')
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].apply(lambda x:x.fillna(x.median()))

# divide th ex & y variables
X = data.drop(['mpg'],axis=1)
y=data[['mpg']]

# scaling the dataset 
X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X_s , columns=X.columns).values
y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns=y.columns).values

# train test split 
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=0)


# simple linear regression model 
regressor = LinearRegression()
regressor.fit(x_train,y_train)

for idx,col_name in enumerate(x_train.columns):
    print('coefficient for {} is {}'.format(col_name , regressor.coef_[0][idx]))


# intercept / constant value
intercept = regressor.intercept_[0]
print('the intercept is {}'.format(intercept))


# ridge reguralization model 
ridge_model = Ridge(alpha=0.3)
ridge_model.fit(x_train,y_train)
print('Ridge model coef : {}'.format(ridge_model.coef_))


# Lasso Reguralization model 
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(x_train,y_train)
print('Lasso model coef: {}'.format(lasso_model.coef_))


# score comparision of regressor model , ridge , lasso 
print(regressor.score(x_train,y_train))
print(regressor.score(x_test,y_test))

print('********************************')

print(ridge_model.score(x_train,y_train))
print(ridge_model.score(x_test,y_test))

print('********************************')

print(lasso_model.score(x_train,y_train))
print(lasso_model.score(x_test,y_test))


# model parameter tunning 
data_train_test = pd.concat([x_train , y_train],axis=0)
data_train_test.head()


import statsmodels.api as sm

X = data.drop('mpg', axis=1)
y = data['mpg']

X = X.astype(float)
y = y.astype(float)

X = sm.add_constant(X)

print(X.shape)        # must be (n_samples, >=2)
print(X.columns)      # verify columns


regressor_OLS = sm.OLS(y, X).fit()
print(regressor_OLS.summary())

selected_cols = ['const', 'cyl', 'disp', 'hp', 'wt', 'acc', 'yr', 'car_type','origin_america', 'origin_asia', 'origin_europe']
X_opt = X[selected_cols]

regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())


mse = np.mean((regressor.predict(x_test)-y_test)**2)
import math 
rmse = math.sqrt(mse)
print('Root mean squared error: {}'.format(rmse))

fig = plt.figure(figsize=(10,8))
sns.residplot(x= x_test['hp'], y= y_test['mpg'], color='green', lowess=True )


fig = plt.figure(figsize=(10,8))
sns.residplot(x= x_test['acc'], y= y_test['mpg'], color='green', lowess=True )

y_pred = regressor.predict(x_test)
plt.scatter(y_test['mpg'],y_pred)




