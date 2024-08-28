
#Importing of desired libraries
import numpy as np
import pandas as pd
#----------------------------------Reading the dataset-----------------------
dataset = pd.read_csv('salary.csv')
# pd is the alias of the Pandas library imported.
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values

# handling of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#label encoding categorical attribute city
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
print(X[:, 0])

#One-hot encoding categorical attribute city
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
#Alternate way to do One-hot encoding is mentioned in the end of this file


#label encoding purchased attribute
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#Splitting of Dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn import preprocessing
X_train = preprocessing.normalize(X_train, norm='l1')
X_test = preprocessing.normalize(X_test, norm='l1')

#Standardizing the data
#lets do split again as we have already applied normalization on X_train and X_test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#apply standarization
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train) 
X_test = scale_X.transform(X_test)




#Alternate way to do One-hot encoding
#Importing of desired libraries
import numpy as np
import pandas as pd
#---------------------------Reading the dataset----------------------
dataset = pd.read_csv('salary.csv')
# pd is the alias of the Pandas library imported.
dataset = pd.get_dummies(dataset, columns=["City"], drop_first=True)