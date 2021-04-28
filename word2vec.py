# Importing required packages
import os
import gensim
import json
import pandas as pd
import nltk
import re
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import math
import numpy as np

# Downloading wordnet from NLTK to perform Stemmer
nltk.download('wordnet')

#Importing CellPhones and Accessories Amazon Reviews Dataset
reviews=[]
for line in open('C:/Users/anish.goyal/Documents/ampba/Term 6/AAI/Cell_Phones_and_Accessories_5.json'):
    reviews.append(json.loads(line))
	
	
#Creating dataframe with required columns- reviewText and overallrating
reviewDF=pd.DataFrame.from_dict(reviews)
reviewDF=reviewDF[['reviewText','overall']]
reviewDF=reviewDF[reviewDF['reviewText']!='']

#Dividing the dataset into train and test
index=math.ceil(0.8*reviewDF.shape[0])
train=reviewDF.iloc[0:index,:]
test=reviewDF.iloc[index:,:]

#Checking size of training and test dataset
print(train.shape)
print(test.shape)

# Stemmer with Python nltk package
stemmer = nltk.PorterStemmer()

# Download all the stopwords from the NLTK package using nltk.download('stopwords')
nltk.download('stopwords')
from nltk.corpus import stopwords  
stopWords = set(stopwords.words('english'))

#Wrapper function to extract tokens from text after cleaning and stemming
def getTokens(text):
    train_tokens=[]
    for line in text:

        # findall() function returns a list containing all matches between a-z
        words = re.findall(r'(\b[a-z][a-z]*\b)', line.lower()) 

        # Stemming each word in to a list, if the word is not in stopwords
        words = [stemmer.stem(word) for word in words if word not in stopWords]

        train_tokens.append(words)  
    return train_tokens
	
	
#Converting training data reviews to tokens
train_tokens=getTokens(train['reviewText'])

#Building word2vec with size=100
model_100 = Word2Vec(train_tokens, size=100, window =5,min_count=5)

vectors=model_100.wv

vectors.similar_by_word('work')

vectors.similar_by_word('shape')

model_300 = Word2Vec(train_tokens, size=300, window =5,min_count=5) 

vectors=model_300.wv
vectors.similar_by_word('work')

vectors.similar_by_word('shape')

def getAvgWordEmbedding(doc):
    doc = [word for word in doc if word in model_100.wv.vocab]
    try:
        return np.mean(model_100[doc], axis=0)
    except:
        return ''
		
train['vectorList']=[getAvgWordEmbedding(tlist) for tlist in train_tokens]
test['vectorList']=[getAvgWordEmbedding(tlist) for tlist in getTokens(test['reviewText'])]

train_filtered=train[train['vectorList']!='']
test_filtered=test[test['vectorList']!='']

lb = LabelEncoder()
y_train = lb.fit_transform(train_filtered['overall'])
lb1 = LabelEncoder()
y_test = lb1.fit_transform(test_filtered['overall'])

X_train=list(train_filtered['vectorList'])
X_test=list(test_filtered['vectorList'])
y_train=list(train_filtered['overall'])
y_test=list(test_filtered['overall'])

#X_train = np.asarray(train['vectorList'])
#X_test = list(test['vectorList'])
#X_train = pd.DataFrame(train['vectorList'],columns=['f'+str(i) for i in range(0,100)])
clf = LogisticRegression(C=100)

clf.fit(X_train, y_train)

y_pred_train=clf.predict(X_train)
y_pred = clf.predict(X_test)
#y_pred = lb.inverse_transform(y_test)

accuracy_score(y_train,y_pred_train)

accuracy_score(y_test,y_pred)

def getAvgWordEmbedding(doc):
    doc = [word for word in doc if word in model_100.wv.vocab]
    try:
        return np.mean(model_300[doc], axis=0)
    except:
        return ''
		
train['vectorList']=[getAvgWordEmbedding(tlist) for tlist in train_tokens]
test['vectorList']=[getAvgWordEmbedding(tlist) for tlist in getTokens(test['reviewText'])]


train_filtered300=train[train['vectorList']!='']
test_filtered300=test[test['vectorList']!='']

lb = LabelEncoder()
y_train = lb.fit_transform(train_filtered300['overall'])
lb1 = LabelEncoder()
y_test = lb1.fit_transform(test_filtered300['overall'])

X_train=list(train_filtered300['vectorList'])
X_test=list(test_filtered300['vectorList'])
y_train=list(train_filtered300['overall'])
y_test=list(test_filtered300['overall'])

clf = LogisticRegression(C=100)

clf.fit(X_train, y_train)

y_pred_train=clf.predict(X_train)
y_pred = clf.predict(X_test)

accuracy_score(y_train,y_pred_train)
accuracy_score(y_test,y_pred)