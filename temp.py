#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the datasets
dataset=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus=[]
for i in range(0,7613):
    review=re.sub('[^a-zA-Z]' , ' ',dataset['text'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10766)
X_train=cv.fit_transform(corpus).toarray()
y_train=dataset.iloc[:,4].values

#splitting the dataset into training and test set
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#fitting naive bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)



#making the confusion matrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test, y_pred)

#sample_submission=pd.read_csv()


#
corpus1=[]
for j in range(0,3263):
    review1=re.sub('[^a-zA-Z]' , ' ',dataset2['text'][j])
    review1=review1.lower()
    review1=review1.split()
    ps1=PorterStemmer()
    review1=[ps1.stem(word) for word in review1 if not word in stopwords.words('english')]
    review1 = ' '.join(review1)
    corpus1.append(review1)

cv1=CountVectorizer()
X_test=cv1.fit_transform(corpus1).toarray()


#predicting the results
y_pred=classifier.predict(X_test)


sample_submission=pd.read_csv('sample_submission.csv')
sample_submission["target"]=y_pred
sample_submission.to_csv("submission.csv", index=False)