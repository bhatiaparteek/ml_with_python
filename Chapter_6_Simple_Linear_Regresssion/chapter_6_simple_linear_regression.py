#"""Chapter 6: - Simple Linear Regression

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the Dataset
dataset = pd.read_csv('StipendData.csv')

# Creating X and y
X = dataset.iloc[:, :- 1].values
y = dataset.iloc[:, 1].values

#Performing train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 1/3, random_state = 0)

#Building the model
#importing LinearRegression class
from sklearn.linear_model import LinearRegression
#creating object
linear_regressor = LinearRegression()

#Fitting the model
linear_regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = linear_regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linear_regressor.predict(X_train), color = 'blue')
plt.title('Visualization of Training Data')
plt.xlabel('Years of Research Experience')
plt.ylabel('Stipend')
plt.show()

#Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, linear_regressor.predict(X_test), color = 'blue')
plt.title('Visualization of Test Data')
plt.xlabel('Years of Experience')
plt.ylabel('Stipend')
plt.show()

#Performance Evaluation
# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the results using MAE
print(metrics.mean_absolute_error(y_test, y_pred))
# Evaluating the model and printing the results using MSE
print(metrics.mean_squared_error(y_test, y_pred))
# Evaluating the model and printing the results using RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Evaluating the model and printing the results using R2
print(metrics.r2_score(y_test, y_pred))



