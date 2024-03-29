# -*- coding: utf-8 -*-
"""Chapter 17 Implementation of Artificial Neural Network

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bw05SSvfIYLmV_h91PG6b7JxOZl7Ju4k

1. Importing the Libraries
"""

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

"""2. Importing the Data"""

#Importing the Data
dataset = pd.read_csv('MallCustomerDataset_.csv')

"""3. Feature and Label selection"""

#Feature and Label selection
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

"""4. Encoding categorical data

"""

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

"""5. Splitting the dataset into the Training set and Test set 


"""

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""6. Feature Scaling 

"""

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

"""7. Importing the ANN Libraries

"""

#Importing the ANN Libraries
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout

"""8. Intialising the ANN

"""

#Intialising the ANN
classifier = Sequential()

"""9. Adding the input layer and the first hidden layer"""

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation = "relu", input_dim = 30, units=16,  kernel_initializer = "uniform"))
 # Adding the input layer and the first hidden layer
classifier.add(Dropout(rate=0.1))

"""10. Adding the second hidden layer

"""

# Adding the second hidden layer
classifier.add(Dense(activation = "relu", units=16,  kernel_initializer  = "uniform"))
 # Adding dropout to prevent overfitting
classifier.add(Dropout(rate=0.1))

"""11.  Adding the output layer



"""

# Adding the output layer
classifier.add(Dense(activation = "sigmoid", units=1, kernel_initializer = "uniform"))

"""12. Compining the ANN

"""

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
