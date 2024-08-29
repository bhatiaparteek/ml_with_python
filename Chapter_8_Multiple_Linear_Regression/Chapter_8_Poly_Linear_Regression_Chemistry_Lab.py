# Polynomial Regression Case Study of Chemistry Lab Experiment
#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Step 1: Load the dataset"""
#Loading the dataset
Dataset = pd.read_csv('Chemistry_Lab.csv')  

# Creating Data Frames
X = Dataset.iloc[:, 1: 2].values
Y = Dataset.iloc[:, 2].values  

"""Step 2: Splitting the dataset into training and testing datasets"""
#No need in this case, as dataset is very small.

"""Step 3: Transforming X to higher degrees for building the polynomial regression model"""
#Transforming X to higher degree polynomial terms 
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 2)
X_poly = Poly_reg.fit_transform(X)

"""Step 4: Fitting the polynomial regression model"""
#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, Y) 

"""Step 5: Visualizing the polynomial regression results"""
# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')  
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

"""Step 6: Performance evaluation for polynomial regression with degree 2"""
#Getting predicted values for Polynomial Regression with degree 2
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

""Step 7: Changing the degree of polynomial to 3"""
#Doing experiment by transfroming the X to degree 3

#Transforming X to higher degree polynomial terms 
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 3)
X_poly = Poly_reg.fit_transform(X)

#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, Y) 

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')  
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()# Visualising the Polynomial Regression results

#Getting predicted values for Polynomial Regression with degree 3
Y_Pred_3=Poly_reg_model.predict(X_poly)

#Performance Evaluation for Polynomial Regression with degree 3
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error with Degree 3')
print(metrics.mean_absolute_error(Y, Y_Pred_3))

# Evaluating the model and printing the value of MSE
print('Mean Square Error with Degree 3')
print(metrics.mean_squared_error(Y, Y_Pred_3))

# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error with Degree 3')
print(np.sqrt(metrics.mean_squared_error(Y, Y_Pred_3)))

""""Step 8: Predict polynomial regression results"""
#Predciting the value of pressure at temperarure 30 by using polynomial regreession
Poly_reg_model.predict(Poly_reg.fit_transform([[30]]))

"""Step 9: Comparing the results of polynomial regression with the results of simple linear
regression"""
# Training the Linear Regression model on the whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y) 

# Visualizing the Simple Linear Regression results
plt.scatter(X, Y, color = 'red') 
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show() 

#Getting predicted values for simple linear regression
Y_Pred_linear=linear_regressor.predict(X)

#Performance Evaluation for simple linear regression
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error with simple linear regression')
print(metrics.mean_absolute_error(Y, Y_Pred_linear))

# Evaluating the model and printing the value of MSE
print('Mean Square Error with simple linear regression')
print(metrics.mean_squared_error(Y, Y_Pred_linear))

# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error with simple linear regression')
print(np.sqrt(metrics.mean_squared_error(Y, Y_Pred_linear)))

#Predciting the value of pressure at temperarure 30 by using polynomial regreession
linear_regressor.predict([[30]])
 