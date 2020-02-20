#https://github.com/coosto/dutch-word-embeddings
import pandas as pd
import gensim

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



db = positive.append(negative)
db['target'] = db['search_string']
db.target.value_counts()


ngramrange = (1,3)




#gensim pretrained use example https://medium.com/cindicator/t-sne-and-word-embedding-weekend-of-a-data-scientist-5c99ddacbf51
model_file_folder = 'C:/Users/Michael/Statistiek Vlaanderen/hlmos-statistiek-vlaanderen-twitter/dutch-word-embeddings/'
model_file_name = 'model.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file_folder+model_file_name, binary=True)


