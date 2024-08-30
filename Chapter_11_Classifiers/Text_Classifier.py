# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:47:04 2022

@author: Parteek Bhatia
"""

#Importing the libraries 
import pandas as pd
import re
import nltk
import nltk
nltk.download('stopwords') #downloading english stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#reading the dataset
dataset=pd.read_csv('IMDB Dataset.csv')
x=dataset.iloc[:,0].values #string review text in x
y=dataset.iloc[:,1].values #string review class in y
reviews = [] #creating an empty list to store a review
corpus = []#creating an empty list to store corpus containing all the reviews

#string review text for only 100 rows in x
x=dataset.iloc[0:100,0].values
#string review class for only 100 rows in y
y=dataset.iloc[0:100,1].values


#processing of first 100 reviews in for loop
for character in range(0, len(x)):
    """
    processing of all reviews in for loop removing any special
    characters or numbers from reviews if word is not containing
    alphabets then it will be replaced by blank space otherwise
    passed as such
    """
    review = re.sub('[^a-zA-Z]', ' ', str(x[character]))
    """
    [^a-zA-Z] means any character that IS NOT a-z OR A-Z sub
    method return the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in string by the
    replacement mentioned as second parameter
    """
    # converting the text to lower case
    review = review.lower()
    #split the sentences into words by split it over blank space
    review = review.split()
    """
    extracting the root words and removing the stop words from
    the word in the review
    """
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    """
    processed words are joined to form a review text
    all processed reviews are appended into a corpus
    """
    corpus.append(review)
    # \ indicates continuation of the code in next line
    # Multiline comments are indicated with triple double quotes
    #(""") at the beginning and end of the comment block.
    
# Applying Count Vectorizer for converting strings to vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()


# Applying TF-IDF for converting strings to vector
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Fitting RandomForest to the Training set
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators=1200, random_state=35)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
