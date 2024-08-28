
#1. Importing the Libraries

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""2. Data Acquistion"""
dataset = pd.read_csv('50_AdAgency.csv')

#"""3.  Creating Data Frames
# Creating Data Frames
X = dataset.iloc[:, :- 1].values
Y = dataset.iloc[:, 4].values

#Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])


#One-hot encoding categorical attribute city
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

"""5. Avoiding the Dummy Variable Trap"""

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

"""6. Dataset Split"""
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, random_state = 0)

"""7. Model building"""
#building the linear model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)

"""8. Predictions"""
# Predicting the Test set results
Y_pred = linear_regressor.predict(X_test)

# Model Evaluation
# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error')
print(metrics.mean_absolute_error(Y_test, Y_pred))
# Evaluating the model and printing the value of MSE
print('Mean Square Error')
print(metrics.mean_squared_error(Y_test, Y_pred))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error')
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

