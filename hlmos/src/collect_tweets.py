import tweepy
import pandas as pd
import datetime as DT
import time
##
## Connect to Twitter API
##

with open('../keys') as f:
    keys = f.readlines()
keys = [x.strip() for x in keys]
consumer_key = keys[0]
consumer_secret = keys[1]
access_token  = keys[2]
access_token_secret = keys[3]



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)




##
## Collect tweets function and save them
##
def collect_tweets(search, keyword, location=None, location_granularity = 'country',lang='nl', result_type='mixed', limit=0, retweets=False):
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=7)
    week_ago = week_ago.strftime("%Y-%m-%d")
    date_since = week_ago

    q = keyword



    search_query = "{0}".format(keyword)
    if location != None:
        places = api.geo_search(query=location, granularity=location_granularity)
        place_id = places[0].id
        search_query = search_query+"&place:{1}".format(place_id)


    if not retweets:
        search_query = search_query + " -filter:retweets"
    tweets = tweepy.Cursor(search, q=search_query, lang=lang, tweet_mode='extended',since=date_since)
    tweets = tweets.items()
    x = []
    counter = 0
    while True:
        #TODO add code that flushed tweets do disk every X tweets (for very large collections)
        try:
            tweet = tweets.next()
            print(counter)
            if counter > limit:
                break
            counter = counter + 1
            x.append([tweet.id_str, keyword, tweet.full_text, tweet.created_at, tweet.lang, tweet.retweeted,
                      tweet.author.screen_name, tweet.author.name, tweet.source])
        except tweepy.TweepError:
            print('sleeping')
            time.sleep(60 * 15)
            continue
        except StopIteration:
            break

    x = pd.DataFrame.from_records(x, columns=["doc_id", "search_string", "text", "created_at", "language", "is_retweet",
                                              "author", "author_name", "source"])
    return (x)


## A. data collection based on emoticons

nb_tweets_to_collect = 20000

positive_emoji = [":)", ":-)", ":D", ":-D", ": )"]
positive_search_string = " OR ".join(positive_emoji)

positive_search_string = positive_emoji[0]
x = collect_tweets(api.search, keyword=positive_search_string, location=None, location_granularity = 'country', lang="nl", result_type="mixed", limit=nb_tweets_to_collect)
x.to_csv("tweets_positive"+DT.datetime.utcnow().strftime('%Y%m%d %H%M%S%f')+".csv", encoding='utf-8', index=False, sep=";")
# x.to_pickle("tweets_positive.pck", compression=None)


negative_emoji = [":(", ": (", ":'("]
negative_search_string = " OR ".join(negative_emoji)
x = collect_tweets(api.search, keyword=negative_search_string, location=None, location_granularity = 'country', lang="nl", result_type="mixed", limit=nb_tweets_to_collect)
x.to_csv("tweets_negative"+DT.datetime.utcnow().strftime('%Y%m%d %H%M%S%f')+".csv", encoding='utf-8', index=False, sep=";")
# x.to_pickle("tweets_negative.pck", compression=None)