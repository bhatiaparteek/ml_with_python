"""1.1: Importing the Libraries"""

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""1.2: Loading the Dataset"""

#Reading the dataset
Dataset = pd.read_csv('Alexa_dataset.csv')

#Creating X and Y
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values


"""1.3: Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""1.4: Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

"""1.5: Fitting Decision Tree Classification to the Training set """

""" Classifier – 1 (Decision Tree Classifier)"""


# Fitting Decision Tree Classification to the Training set 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

"""1.6:	Predicting the Decision Tree Classifier"""

# Predicting the Decision Tree Classifier
y_pred = classifier.predict(X_test)

"""1.7: Making the confusion matrix"""

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""1.8: Performance metrics"""

#Performance metrics
# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is ', accuracy)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is ', precision)
 
# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is ', recall)

"""1.9: Function definition for visualization of results"""

# Function definition for visualization of results
def Visualizer(argument1, arguement2):
  from matplotlib.colors import ListedColormap
  X_set, y_set = argument1, arguement2
  X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, \
           stop=X_set[:,0].max()+1, step=0.01), \
           np.arange(start=X_set[:, 1].min() - 1, \
           stop = X_set[:, 1].max() + 1, step = 0.01))
  plt.contourf(X1,X2, \
  classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), \
                     alpha= 0.75, cmap = ListedColormap(('red', 'green')))
  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())
  for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label=j)
  plt.xlabel('Age')
  plt.ylabel('Minutes_of_Music_Consumed')
  plt.legend()
  plt.show()
#  \ indicates continuation of the code in next line

"""1.10: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""1.11: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)




"""Classifier – 2 (Random Forest Classifier)

2.1: Importing the Libraries """

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""2.2: Loading the Dataset"""

#Loading the dataset
Dataset = pd.read_csv('Alexa_dataset.csv')
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values

"""2.3: Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""2.4: Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

"""2.5: Fitting Random Forest Classification to the Training set """

# Classifier – II (Random Forest Classifier)
# Fitting Random Forest Classification to the Training set 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion =  'entropy', random_state = 0)
classifier.fit(X_train, y_train)

"""2.6: Predicting the Test set results"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""2.7: Making the confusion matrix """

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""2.8: Performance metrics"""

#Performance metrics
#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is ', accuracy)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is ', precision)

#Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is ', recall)

"""2.9: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""2.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)




"""Classifier – 3 (Naïve Bayes Classifier)

3.1: Importing the Libraries """

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""3.2: Loading the Dataset"""

#Importing the Dataset
Dataset = pd.read_csv('Alexa_dataset.csv')
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values

"""3.3: Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""3.4: Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

"""3.5: Fitting Naive Bayes to the Training set """

# Classifier – III  (Naïve Bayes Classifier)
# Fitting Naïve Bayes to the Training set 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

"""3.6: Predicting the Test set results"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""3.7: Making the confusion matrix"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""3.8: Performance metrics"""

#Performance metrics
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
print('Recall is ', recall)

"""3.9: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""3.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)



"""Classifier – 4 (K-Nearest Neighbors)

4.1: Importing the Libraries"""

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""4.2: Loading the Dataset"""

#Loading the Dataset
Dataset =  pd.read_csv('Alexa_dataset.csv') 
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values

"""4.3: Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""4.4: Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

"""4.5: Fitting KNN to the Training set """

# Classifier – IV  (KNN)
# Fitting KNN to the Training set 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier( n_neighbors = 5, metric =  'minkowski', p = 2)
classifier.fit(X_train, y_train)

"""4.6: Predicting the Test set results"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""4.7: Making the confusion matrix"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""4.8: Performance metrics"""

#Performance metrics
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
print('Recall is ', recall)

"""4.9: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""4.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)




"""Classifier – 5 (Logistic Regression)

5.1: Importing the Libraries """

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""5.2: Importing the Dataset"""

#Importing the Dataset
Dataset =  pd.read_csv('Alexa_dataset.csv') 
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values

"""5.3: Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""5.4: Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

"""5.5: Fitting LR to the Training set """

# Classifier – V (Logistic Regression)
# Fitting Logistic Regression to the Training set 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression( random_state = 0)
classifier.fit(X_train, y_train)

"""5.6: Predicting the Test set results"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""5.7: Making the confusion matrix """

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""5.8: Performance metrics"""

#Performance metrics

#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is ', accuracy)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is ', precision)

#Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is ', recall)


"""5.9: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""5.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)


"""Classifier – 6 (SVM Linear)

6.1: Importing the Libraries"""

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""6.2: Loading the Dataset"""

#Loading the Dataset
Dataset =  pd.read_csv('Alexa_dataset.csv') 
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values

"""6.3: Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""6.4: Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)


"""6.5: Fitting SVM (Linear) to the Training set """

# Classifier –  VI (SVM - Linear)
# Fitting SVM to the Training set 
from sklearn.svm import SVC
classifier = SVC( kernel =  'linear', random_state = 0)
classifier.fit(X_train, y_train)

"""6.6: Predicting the Test set results"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""6.7: Making the confusion matrix """

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""6.8: Performance metrics"""

#Performance metrics
#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is', accuracy)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is', precision)

#Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is', recall)

"""6.9: Visualizing the Training set results """

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""6.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)





"""Classifier – 7 (SVM Non-Linear)

7.1: Importing the Libraries"""

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""7.2: Loading the Dataset"""

#Loading the Dataset
Dataset =  pd.read_csv('Alexa_dataset.csv') 
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values

"""7.3: Splitting the dataset into the Training set and Test set """

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""7.4: Feature Scaling """

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

"""7.5: Fitting SVM (Non - Linear) to the Training set """

# Classifier – VII (Non-Linear SVM)
# Fitting Non-Linear SVM to the Training set 
from sklearn.svm import SVC
classifier = SVC (kernel =  'rbf', random_state = 0)
classifier.fit(X_train, y_train)

"""7.6: Predicting the Test set results"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""7.7: Making the confusion matrix """

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""7.8: Performance metrics"""

#Performance metrics
#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is', accuracy)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is', precision)

#Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is', recall)

"""7.9: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""7.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)
