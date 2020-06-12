# hlmos-statistiek-vlaanderen-twitter
This repository contains the code written of the HLG-MOS project on Sentiment Analysis of Flemihs tweets, performed by Statistic Flanders.

The code for the project can be found under /hlmos/src . Please be aware that this project is still under experimental development.
For questions about the code, contact michael.reusens@vlaanderen.be

The code is split up in 4 python files. We will discuss their functionality.

## collect_tweets.py
This python script can be run to collect Tweets using the tweepy API. Within the code you can set the parameters related to the number tweets you wish to retrieve, as well as the keywords you are filtering on.
## count and tfidf sentiment models.py
The script reads in the collected tweets (the result of the collect_tweets.py script) and trains + optimizes logistic regression models on both count-vectorized and tfidf-vectorized tweets.
## pretrained_embedding_model.py 
The script reads in the collected tweets (the result of the collect_tweets.py script) and trains + optimizes an XGBoost classifier on tweets that are embedded using a pretrained embedding ("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
## self trained embedding.py
The script reads in the collected tweets (the result of the collect_tweets.py script) and trains + optimizes a classifier on tweets that are embedded using a self-trained auto-encoder.

