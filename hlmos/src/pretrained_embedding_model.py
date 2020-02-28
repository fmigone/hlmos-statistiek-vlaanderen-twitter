#https://github.com/coosto/dutch-word-embeddings
import pandas as pd
import gensim

tweets_file_location=  'C:/Users/Michael/Statistiek Vlaanderen/hlmos-statistiek-vlaanderen-twitter/hlmos/src/'
#read in tweets
negative = pd.read_csv(tweets_file_location+'tweets_negative7k.csv', sep = ";")
positive = pd.read_csv(tweets_file_location+'tweets_positive20k.csv', sep = ";")


# https://realpython.com/python-keras-text-classification/
import datetime as DT
import matplotlib.pyplot as plt



# tweets_file_location=  'C:/Users/Michael/Statistiek Vlaanderen/hlmos-statistiek-vlaanderen-twitter/hlmos/src/'
#read in tweets
# negative = pd.read_csv(tweets_file_location+'tweets_negative7k.csv', sep = ";")
negative['target'] = 0
# positive = pd.read_csv(tweets_file_location+'tweets_positive20k.csv', sep = ";")
positive['target'] = 1

print(len(positive))
print(len(negative))

#Ideally we put these in a file to be read in both be the tweet collection code and the sentiment modelling code to reduce chance of inconsistenties
positive_emoji = [":)", ":-)", ":D", ":-D", ": )"]
negative_emoji = [":(", ": (", ":'("]
all_emoji = positive_emoji+negative_emoji

#Uses Bert on tweets
# https://towardsdatascience.com/russian-troll-tweets-classification-using-bert-abec09e43558


db = positive.append(negative)
db.target.value_counts()

ngramrange = (1,3)


##
## Make a penalised logistic regression model
##
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

txt = db.text.tolist()
clean_texts = [tokenize(text) for text in txt]
clean_texts = [" ".join(text) for text in clean_texts]
db['clean_text'] = clean_texts

y = db.target.values



# https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import numpy as np


import tensorflow as tf
import tensorflow_hub as hub
# tf.contrib.resampler
# !pip install tensorflow-text
import tensorflow_text as text

#load pretrained tensorflow hub code
module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
embed = hub.load(module_url)

#Embed texts using the pretrained model and convert to numpy array
embedded_clean_texts = [embed(text).numpy()[0] for text in clean_texts]

#Convert list to array
embedded_clean_texts = np.array(embedded_clean_texts)

## Split dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(embedded_clean_texts, db.target.values,
                                                    train_size=0.75, random_state=1234)

# optimizing parameter C of penalized logistic regression
# gridsearch voor C (=10 in last try)
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

log_model = LogisticRegression(C=10, penalty="l1", solver="liblinear")
log_model = log_model.fit(X=X_train, y=y_train)
log_model.get_params()
log_model.coef_


## Hoe goed werkte dit model op de test data
y_pred = log_model.predict(X_test)
print("Testing the testing dataset accuracy...")
print(classification_report(y_test, y_pred))


# #Lets try XGB
import xgboost as xgb

#Find best parameterization
param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 2, 5],
        'subsample': [0.6, 1.0],
        'colsample_bytree': [0.6, 1.0],
        'max_depth': [3, 5]
        }

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model_cv = GridSearchCV(xgb_model, param_grid, cv=3,verbose = 5)

xgb_model_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned xgboost parameters Parameters: {}".format(xgb_model_cv.best_params_))
print("Best score is {}".format(xgb_model_cv.best_score_))
#Last run had following optimal params:


xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X=X_train, y=y_train)
y_pred = xgb_model.predict(X_test)
print("Testing the testing dataset accuracy...")
print(classification_report(y_test, y_pred))



#Let's try random forest
from sklearn.ensemble import RandomForestRegressor
#TODO GRID SEARCH FOR OPTIMAL PARAMETERS
# Instantiate model
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)
## Hoe goed werkte dit model op de test data
y_pred = rf.predict(X_test)
print("Testing the testing dataset accuracy tfhub pretrained 3/random forest...")
print(classification_report(y_test, y_pred))
