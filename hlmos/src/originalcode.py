# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:48:37 2019

@author: Marc
"""

# eerst "pip install tweepy" uitvoeren in de command line van anaconda prompt
# of eerst !pip install in IPhyton console

import tweepy
import pandas as pd

##
## Connect to Twitter API
##
consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'


##TODO ADD CODE TO READ FROM FILE (THAT IS GITINGORED!)


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)




##
## Collect tweets function and save them
##
def collect_tweets(search, keyword, location='belgium', location_granularity = 'country',lang='nl', result_type='mixed', limit=0, retweets=False):
    q = keyword

    places = api.geo_search(query=location, granularity=location_granularity)
    place_id = places[0].id

    search_query = "{0}&place:{1}".format(keyword, place_id)

    if not retweets:
        search_query = search_query + " -filter:retweets"
    tweets = tweepy.Cursor(search, q=search_query, count=100, lang=lang, result_type=result_type, tweet_mode='extended')
    tweets = tweets.items(limit)
    x = []
    for tweet in tweets:
        x.append([tweet.id_str, keyword, tweet.full_text, tweet.created_at, tweet.lang, tweet.retweeted,
                  tweet.author.screen_name, tweet.author.name, tweet.source])

    x = pd.DataFrame.from_records(x, columns=["doc_id", "search_string", "text", "created_at", "language", "is_retweet",
                                              "author", "author_name", "source"])
    return (x)


## A. data collection based on emoticons

x = collect_tweets(api.search, keyword=":)", location='belgium', location_granularity = 'country', lang="nl", result_type="mixed", limit=5000)
x.to_csv("tweets_positive.csv", encoding='utf-8', index=False, sep=";")
x.to_pickle("tweets_positive.pck", compression=None)

x = collect_tweets(api.search, keyword=":(", location='belgium', location_granularity = 'country', lang="nl", result_type="mixed", limit=5000)
x.to_csv("tweets_negative.csv", encoding='utf-8', index=False, sep=";")
x.to_pickle("tweets_negative.pck", compression=None)

## X. data collection based on emoticons

x = collect_tweets(api.search, keyword="vlaanderen", lang="nl", result_type="mixed",
                   limit=100, retweets=False)
y = x[~x['text'].str.contains("RT @")]
y = x[~x['text'].str.startswith("RT @")]
x.to_csv("tweets_vla.csv", encoding='utf-8', index=False, sep=";")
x.to_pickle("tweets_vla.pck", compression=None)

x = collect_tweets(api.search, keyword=":(", lang="nl", result_type="mixed", limit=5000)
x.to_csv("tweets_negative.csv", encoding='utf-8', index=False, sep=";")
x.to_pickle("tweets_negative.pck", compression=None)

## B. data collection based on general tweets

##### here ends data collection

#################################
##### Part II here starts processing
##################################
positive = pd.read_pickle("tweets_positive.pck")
negative = pd.read_pickle("tweets_negative.pck")

db = positive.append(negative)
db['target'] = db['search_string']
db.target.value_counts()

db

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

## Maak een document/term/matrix
vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize, lowercase=True,
                             stop_words=[':(', ':)'], min_df=0.001)

txt = db.text.tolist()
dtm = vectorizer.fit_transform(txt)
dtm_nd = dtm.toarray()
dtm_nd.shape
vocab = vectorizer.get_feature_names()

## Maak een logistische regressie en zie naar de parameters van het model
X_train, X_test, y_train, y_test = train_test_split(dtm_nd, db.target,
                                                    train_size=0.75, random_state=1234)
log_model = LogisticRegression(C=1, penalty="l1")
log_model = log_model.fit(X=dtm_nd, y=db['target'])
log_model.get_params()
log_model.coef_

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

log_model = LogisticRegression(C=5, penalty="l1", solver="liblinear")
log_model = log_model.fit(X=X_train, y=y_train)
log_model.get_params()
log_model.coef_

## Hoe goed werkte dit model op de test data
y_pred = log_model.predict(X_test)
print("Testing the testing dataset accuracy...")
print(classification_report(y_test, y_pred))

y_pred

y_test

vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize, lowercase=True,
                             stop_words=[':(', ':)'], vocabulary=vocab)
txt = ["een nieuw tweet", "ja zuper"]
dtm = vectorizer.fit_transform(txt)
log_model.predict(dtm)

vocab
