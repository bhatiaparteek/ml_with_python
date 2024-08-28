# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:18:18 2021

@author: Parteek Bhatia
"""
"""#**Polynomial Regression**

1: Importing of the Libraries
"""

# Polynomial Regression

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""2: Data Acquistion"""


File = 'Position_Salaries.csv'
dataset = pd.read_csv(File)


"""3:  Creating Data Frames

"""

#Creating X and Y
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


"""4: Creating Polynomial Regressor"""

#Transforming X to higher degree polynomial terms
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 2)
X_poly = Poly_reg.fit_transform(X)

"""5: Training the Polynomial Regression model on the whole dataset

"""

#Training the polynomial regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, Y)

"""6: Visualising the Training set results"""

# Visualising the Training set results
#Visualizing the polynomial regression results
plt.scatter(X, Y, color = 'red')
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')
plt.show()


# #Getting predicted values for Polynomial Regression with degree 2
Y_Pred_2=Poly_reg_model.predict(X_poly)
#Performance Evaluation for Polynomial Regression with degree 2
# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error with Degree 2')
print(metrics.mean_absolute_error(Y, Y_Pred_2))
# Evaluating the model and printing the value of MSE
print('Mean Square Error with Degree 2')
print(metrics.mean_squared_error(Y, Y_Pred_2))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error with Degree 2')
print(np.sqrt(metrics.mean_squared_error(Y, Y_Pred_2)))

"""7:  Changing the degree of polynomial to 3 """

#Transforming X to higher degree polynomial terms
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 3)
X_poly = Poly_reg.fit_transform(X)
 
#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, Y)

# Visualising the Polynomial Regression results
plt.scatter(X,Y, color = 'red') 
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')

# #Getting predicted values for Polynomial Regression with degree 3
Y_Pred_3=Poly_reg_model.predict(X_poly)
#Performance Evaluation for Polynomial Regression with degree 3
# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error with Degree 3')
print(metrics.mean_absolute_error(Y, Y_Pred_3))
# Evaluating the model and printing the value of MSE
print('Mean Square Error with Degree 3')
print(metrics.mean_squared_error(Y, Y_Pred_3))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error with Degree 3')
print(np.sqrt(metrics.mean_squared_error(Y, Y_Pred_3)))

"""8: Changing the degree of polynomial to 4"""

#Transforming X to higher degree polynomial terms
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 4)
X_poly = Poly_reg.fit_transform(X)


#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, Y)

# Visualising the Polynomial Regression results
plt.scatter(X,Y, color = 'red') 
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')


 # #Getting predicted values for Polynomial Regression with degree 4
Y_Pred_4=Poly_reg_model.predict(X_poly)
#Performance Evaluation for Polynomial Regression with degree 4
# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error with Degree 4')
print(metrics.mean_absolute_error(Y, Y_Pred_4))
# Evaluating the model and printing the value of MSE
print('Mean Square Error with Degree 4')
print(metrics.mean_squared_error(Y, Y_Pred_4))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error with Degree 4')
print(np.sqrt(metrics.mean_squared_error(Y, Y_Pred_4)))

"""9: Making predcitions on the best trained model, the model with degree 4"""

#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, Y)

Poly_reg_model.predict(Poly_reg.fit_transform([[7.5]]))


"""10: What will Happen if You Increase the Degree Value Too Much?"""

#Transforming X to degree 50
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 50)
X_poly = poly_reg.fit_transform(X)

#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, Y)

# Visualising the Polynomial Regression results
plt.scatter(X,Y, color = 'red') 
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')


"""11: Comparison of Linear Regression and Polynomial Regression Results"""

#Comparison of Linear Regression and Polynomial Regression Results
from sklearn.linear_model import LinearRegression
# Training the Linear Regression model on the whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)


# Visualising the Linear Regression results
plt.scatter(X,Y, color = 'red') 
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.show()


# Getting predicted values for simple linear regression
Y_Pred_linear= linear_regressor.predict(X)


#Performance Evaluation for simple linear regression
# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error for simple linear regression ')
print(metrics.mean_absolute_error(Y, Y_Pred_linear))
# Evaluating the model and printing the value of MSE
print('Mean Square Error for simple linear regression ')
print(metrics.mean_squared_error(Y, Y_Pred_linear))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error for simple linear regression ')
print(np.sqrt(metrics.mean_squared_error(Y, Y_Pred_linear)))


#Predicting a new result with simple linear regression
linear_regressor.predict([[7.5]])
