"""1. Importing the Libraries"""

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

"""2. Importing the Data"""

#Importing the Data
dataset = pd.read_csv('data.csv')

"""3. Feature and Label selection"""

#Feature and Label selection
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

"""4. Encoding categorical data"""

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

"""5. Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

"""6. Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

""" Install keras by running following commands in Console
!pip install keras 
!pip install tensorflow
"""

"""7. Importing the ANN Libraries """

#Importing the ANN Libraries
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout

"""8. Intialising the ANN """

#Intialising the ANN
classifier = Sequential()

"""9. Adding the input layer and the first hidden layer"""

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=16, kernel_initializer = "uniform", activation = "relu", input_dim = 30))

"""10. Adding the second and third hidden layer """

# Adding the second hidden layer
classifier.add(Dense(units=16, kernel_initializer = "uniform", activation = "relu"))

# Adding the third hidden layer
classifier.add(Dense(units=16, kernel_initializer = "uniform", activation = "relu"))

"""11.  Adding the output layer """

# Adding the output layer
classifier.add(Dense(activation = "sigmoid", units=1, kernel_initializer = "uniform"))

# Adding dropout to prevent overfitting
classifier.add(Dropout(rate=0.1))

"""12. Compining the ANN """

# Compining the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

"""13. Fitting the ANN to the Training set"""

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 150)

"""
14. Predicting the Test set results
"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

"""15. Making the confusion Matrix"""

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""16. Evaluating Performance"""

# Performance metrics
# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is ', accuracy)
# Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is ', precision)
# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is ' , recall)