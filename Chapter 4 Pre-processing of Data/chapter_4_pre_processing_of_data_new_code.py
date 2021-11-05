import numpy as np
import pandas as pd

# style 1
import matplotlib.pyplot
from sklearn.preprocessing import StandardScaler
# Used to perform scaling of data under pre-processing phase.

from sklearn.model_selection import train_test_split
# Used to split the dataset into training and testing.

from sklearn.linear_model import LinearRegression
# Used to perform Linear regression.

from sklearn.metrics import confusion_matrix
# Used to perform performance analysis of classifier by making a confusion matrix.

"""2. Data Acquisition"""


dataset = pd.read_csv('salary.csv')
# pd is the alias of the Pandas library imported.
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values

"""
3. Taking care of missing data

	"""

#Taking care of missing data
from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
Imputer = Imputer.fit(X[:,1:3])
X[:,1:3] = Imputer.transform(X[:,1:3])

"""4. Encoding categorical data

"""

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('country', OneHotEncoder(), [0])])

labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

"""Splitting of Dataset into training and testing

"""

#Splitting of Dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

"""#Feature Scaling"""

#Feature Scaling
from sklearn import preprocessing
X_train = preprocessing.normalize(X_train, norm='l1')
X_test = preprocessing.normalize(X_test, norm='l1')

"""Standardizing the data"""

#Standardizing the data
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train) 
X_test = scale_X.transform(X_test)