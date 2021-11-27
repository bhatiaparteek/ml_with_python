
#1. Importing the Libraries

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""2. Data Acquistion"""

File = '50_AdAgency.csv'
dataset = pd.read_csv(File)

"""3.  Creating Data Frames

"""

# Creating Data Frames
X = dataset.iloc[:, :- 1].values
Y = dataset.iloc[:, 4].values

"""4. Encoding categorical data"""

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Geography", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)


"""5. Avoiding the Dummy Variable Trap"""

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

"""6. Dataset Splition"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size= 1/3, random_state = 0)

"""7. Modeling"""

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

"""8. Predictions"""

# Predicting the Test set results
y_pred = linear_regressor.predict(X_test)

"""9. Performance Evaluation

"""

# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the results using MAE
print(metrics.mean_absolute_error(y_test, y_pred))

# Evaluating the model and printing the results using MSE
print(metrics.mean_squared_error(y_test, y_pred))

# Importing the math library
import math

# Evaluating the model and printing the results using RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

"""#**Polynomial Regression**

2.1: Importing of the Libraries
"""

# Polynomial Regression

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""2.2: Data Acquistion"""


File = 'Position_Salaries.csv'
dataset = pd.read_csv(File)


"""2.3:  Creating Data Frames

"""

# Creating Data Frames
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""2.4: Creating Polynomial Regressor"""

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)

"""2.5: Training the Polynomial Regression model on the whole dataset

"""

#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, y)

"""2.6: Visualising the Training set results"""

# Visualising the Training set results
plt.scatter(X, y, color = 'red')
plt.title('Visualization of Training Data')
plt.xlabel('Years of Research Experience')
plt.ylabel('Stipend')
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')

"""2.7:  Changing the degree of polynomial to 3 """

#Transforming X to higher degree polynomial terms
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 3)
X_poly = Poly_reg.fit_transform(X)
 
#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X,y, color = 'red') 
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')

"""2.8: Changing the degree of polynomial to 4"""

#Transforming X to higher degree polynomial terms
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 4)
X_poly = Poly_reg.fit_transform(X)
 
#Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Poly_reg_model = LinearRegression()
Poly_reg_model.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X,y, color = 'red') 
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.plot(X, Poly_reg_model.predict(X_poly), color = 'blue')

"""2.9: Predict Polynomial Regression Results"""

Poly_reg_model.predict(Poly_reg.fit_transform([[7.5]]))

"""2.10: Comparison of Linear Regression and Polynomial Regression Results"""

#Comparison of Linear Regression and Polynomial Regression Results
from sklearn.linear_model import LinearRegression
# Training the Linear Regression model on the whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

"""2.11: Visualising the Polynomial Regression results"""

# Visualising the Polynomial Regression results
plt.scatter(X,y, color = 'red') 
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()