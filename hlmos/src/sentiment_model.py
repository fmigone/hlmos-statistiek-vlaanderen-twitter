# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:48:37 2019

@author: Marc Callens, Michael Reusens
"""
# eerst "pip install tweepy" uitvoeren in de command line van anaconda prompt
# of eerst !pip install in IPhyton console
import pandas as pd
import datetime as DT



tweets_file_location=  'C:/Users/Michael/Statistiek Vlaanderen/hlmos-statistiek-vlaanderen-twitter/hlmos/src/'
#read in tweets
negative = pd.read_csv(tweets_file_location+'tweets_negative7k.csv', sep = ";")
positive = pd.read_csv(tweets_file_location+'tweets_positive20k.csv', sep = ";")


print(len(positive))
print(len(negative))

#Ideally we put these in a file to be read in both be the tweet collection code and the sentiment modelling code to reduce chance of inconsistenties
positive_emoji = [":)", ":-)", ":D", ":-D", ": )"]
negative_emoji = [":(", ": (", ":'("]
all_emoji = positive_emoji+negative_emoji

#Uses Bert on tweets
# https://towardsdatascience.com/russian-troll-tweets-classification-using-bert-abec09e43558





# ## X. data collection based on emoticons
#
# x = collect_tweets(api.search, keyword="vlaanderen", lang="nl", result_type="mixed",
#                    limit=100, retweets=False)
# y = x[~x['text'].str.contains("RT @")]
# y = x[~x['text'].str.startswith("RT @")]
# x.to_csv("tweets_vla.csv", encoding='utf-8', index=False, sep=";")
# x.to_pickle("tweets_vla.pck", compression=None)
#
# x = collect_tweets(api.search, keyword=":(", lang="nl", result_type="mixed", limit=5000)
# x.to_csv("tweets_negative.csv", encoding='utf-8', index=False, sep=";")
# x.to_pickle("tweets_negative.pck", compression=None)
#
# ## B. data collection based on general tweets
#
# ##### here ends data collection
#
# #################################
# ##### Part II here starts processing
# ##################################
# positive = pd.read_pickle("tweets_positive.pck")
# negative = pd.read_pickle("tweets_negative.pck")
#
db = positive.append(negative)
db['target'] = db['search_string']
db.target.value_counts()

# db

##
## Make a penalised logistic regression model
##
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import DutchStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt')

stemmer = DutchStemmer(ignore_stopwords=False)


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


tokenize('We proberen even de stemmer uit')

#TODO EXCLUDE MORE SMILEYS
## Maak een document/term/matrix
vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize, lowercase=True,
                             stop_words=all_emoji, min_df=0.001)


txt = db.text.tolist()
dtm = vectorizer.fit_transform(txt)
dtm_nd = dtm.toarray()
dtm_nd.shape
vocab = vectorizer.get_feature_names()
term_index_map = pd.DataFrame({'i' : list(vectorizer.vocabulary_.values()), 'term' : list(vectorizer.vocabulary_.keys())})
term_index_map = term_index_map.sort_values(by='i')

## Maak een logistische regressie en zie naar de parameters van het model
X_train, X_test, y_train, y_test = train_test_split(dtm_nd, db.target,
                                                    train_size=0.75, random_state=1234)
log_model = LogisticRegression(C=1, penalty="l1")
log_model = log_model.fit(X=dtm_nd, y=db['target'])
log_model.get_params()



## Hoe goed werkte dit model op de test data
y_pred = log_model.predict(X_test)
print("Testing the testing dataset accuracy...")
print(classification_report(y_test, y_pred))

# optimizing parameter C
# gridsearch voor C (=1)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {'C': np.arange(0.01, 100, 0.25).tolist()}
param_grid = {'C': [0.1, 0.5, 0.75, 0.8, 0.9, 1.1, 1.4, 5, 10, 20, 30]}

log_model = LogisticRegression(penalty="l1", solver="liblinear")
log_model_cv = GridSearchCV(log_model, param_grid, cv=5)

# Fit it to the data
log_model_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(log_model_cv.best_params_))
print("Best score is {}".format(log_model_cv.best_score_))

log_model = LogisticRegression(C=1, penalty="l1", solver="liblinear")
log_model = log_model.fit(X=X_train, y=y_train)
log_model.get_params()
log_model.coef_
term_index_map['coef'] = log_model.coef_[0]

#Inspect which words have highest coefficients
print('top keywords negatief sentiment')
print(term_index_map.sort_values(by='coef', ascending = True).head(20))
print('top keywords positief sentiment')
print(term_index_map.sort_values(by='coef', ascending = False).head(20))
#Inspection shows serious overfitting --> use some dimensionality reduction scheme (such as PCA, neural network embeddings,...)



## Hoe goed werkte dit model op de test data
y_pred = log_model.predict(X_test)
print("Testing the testing dataset accuracy...")
print(classification_report(y_test, y_pred))

y_pred

y_test

vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize, lowercase=True,
                             stop_words=all_emoji, vocabulary=vocab)
txt = ["een nieuw tweet", "ja zuper"]
dtm = vectorizer.fit_transform(txt)
log_model.predict(dtm)

vocab


#Use fasttext pretrained word vectors: https://fasttext.cc/docs/en/crawl-vectors.html