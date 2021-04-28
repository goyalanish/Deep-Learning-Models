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

reviewDF.to_csv('reviewDf.csv',sep='|')

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


model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/anish.goyal/Documents/ampba/Term 6/AAI/model/GoogleNews-vectors-negative300.bin', binary=True)
vocab = model.vocab.keys()

def getAvgWordEmbedding(doc):
    doc = [word for word in doc if word in model.wv.vocab]
    try:
        return np.mean(model[doc], axis=0)
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

GloVe_Dict = {}
# Loading the 100-dimensional vector of the model
with open("C:/Users/anish.goyal/Documents/ampba/Term 6/AAI/model/glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        GloVe_Dict[word] = vector
		
# Length of the word vocabulary
print(len(GloVe_Dict))

import pprint
# Creating a PrettyPrinter() object
pp = pprint.PrettyPrinter()

GloVe_Dict.keys

def getAvgWordEmbedding(doc):
    doc = [word for word in doc]
    try:
        return np.mean(GloVe_Dict[doc], axis=0)
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

clf = LogisticRegression(C=100)

clf.fit(X_train, y_train)

y_pred_train=clf.predict(X_train)
y_pred = clf.predict(X_test)

accuracy_score(y_train,y_pred_train)

accuracy_score(y_test,y_pred)