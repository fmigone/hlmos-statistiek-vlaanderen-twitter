# https://realpython.com/python-keras-text-classification/
import pandas as pd
import datetime as DT
import matplotlib.pyplot as plt



tweets_file_location=  'C:/Users/Michael/Statistiek Vlaanderen/hlmos-statistiek-vlaanderen-twitter/hlmos/src/'
#read in tweets
negative = pd.read_csv(tweets_file_location+'tweets_negative7k.csv', sep = ";")
negative['target'] = -1
positive = pd.read_csv(tweets_file_location+'tweets_positive20k.csv', sep = ";")
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
db['clean_text'] = clean_texts

y = db.target.values


#END OF DATA PREPROCESSING


#BEGINNING OF MODELLING


from keras.preprocessing.text import Tokenizer

sentences = clean_texts

sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


from keras.preprocessing.sequence import pad_sequences

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from keras.models import Sequential
from keras import layers

embedding_dim = 300

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)



# include Bert using pretrained model
# keras bert: https://github.com/CyberZHG/keras-bert
#pretrained dutch tweet embeddings:  https://github.com/coosto/dutch-word-embeddings


#
#
# tokenize('We proberen even de stemmer uit')
#
# ## Maak een document/term/matrix
# vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize, lowercase=True,
#                              stop_words=all_emoji, min_df=0.001, ngram_range = ngramrange)
#
#

# dtm = vectorizer.fit_transform(txt)
# dtm_nd = dtm.toarray()
# dtm_nd.shape
# vocab = vectorizer.get_feature_names()
# term_index_map = pd.DataFrame({'i' : list(vectorizer.vocabulary_.values()), 'term' : list(vectorizer.vocabulary_.keys())})
# term_index_map = term_index_map.sort_values(by='i')
#
# ## Maak een logistische regressie en zie naar de parameters van het model
# X_train, X_test, y_train, y_test = train_test_split(dtm_nd, db.target,
#                                                     train_size=0.75, random_state=1234)
#
#
#
#
# from keras.models import Sequential
# from keras import layers
#
# input_dim = X_train.shape[1]  # Number of features
#
# model = Sequential()
# model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()
#
# history = model.fit(X_train, y_train,
#                     epochs=100,
#                     verbose=False,
#                     validation_data=(X_test, y_test),
#                     batch_size=10)
#
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
#
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#
# def plot_history(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#
#
# plot_history(history)
#
#
# from keras.preprocessing.text import Tokenizer
#
#
#
#
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(sentences_train)
#
# X_train = tokenizer.texts_to_sequences(sentences_train)
# X_test = tokenizer.texts_to_sequences(sentences_test)
#
# vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
#
# print(sentences_train[2])
# print(X_train[2])
