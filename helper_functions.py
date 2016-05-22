# A file to store helper functions

import requests
import simplejson
from requests_oauthlib import OAuth1 # used for twitter's api
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from math import sqrt

# Load and set twitter API authorization
with open("twitter_secrets.json.nogit") as f:
    secrets = simplejson.loads(f.read())

auth = OAuth1(
    secrets["api_key"],
    secrets["api_secret"],
    secrets["access_token"],
    secrets["access_token_secret"]
)


def grab_user_tweets(screen_name, max_tweets=3200, exclude_replies='true', include_rts='false'):
    """
    Load Twitter timeline for the specified user

    :param screen_name: Twitter screen name to process
    :param max_tweets: maximum number of tweets to get. (Default: 3200, the API maximum)
    :param exclude_replies: exclude replies? (Default: true)
    :param include_rts: include retweets? (Default: false)
    :return: pandas DataFrame of tweets
    """

    # API only allows up to 200 tweets per page
    max_pages = int(min(max_tweets, 3200)//200)
    if max_pages < 1:
        max_pages = 1

    # Need to be strings not booleans
    if isinstance(exclude_replies, type(True)):
        exclude_replies = str(exclude_replies).lower()
    if isinstance(include_rts, type(True)):
        include_rts = str(include_rts).lower()

    if max_tweets < 200:
        count = int(max_tweets)
    else:
        count = 200

    page = 0
    url = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
    for i in range(max_pages):
        if page == 0:
            params = {'screen_name': screen_name, 'count': count, 'lang': 'en',
                      'exclude_replies': exclude_replies, 'include_rts': include_rts}
        else:
            max_id = data[-1]['id'] - 1
            params = {'screen_name': screen_name, 'count': count, 'lang': 'en',
                      'exclude_replies': exclude_replies, 'include_rts': include_rts,
                      'max_id': max_id}

        r = requests.get(url, auth=auth, params=params)
        data = simplejson.loads(r.text)

        if page == 0:
            df = json_normalize(data)
        else:
            df = df.append(json_normalize(data), ignore_index=True)

        page += 1

    return df


def print_dtm(dtm, tweet, num):
    """
    Quick function to print out tweet and document term matrix

    :param dtm: document term matrix
    :param tweet: tweet text array
    :param num: number to show
    :return: None
    """
    for key in dtm[num].keys():
        if dtm[num][key] > 0:
            print('{}: {}'.format(key, dtm[num][key]))
    print(tweet[num])

    return


def make_plot(pcarray, labels, xval=0, yval=1):
    plt.figure()
    plt.plot(pcarray.iloc[:, xval][labels == 0], pcarray.iloc[:, yval][labels == 0], 'bo',
             alpha=0.6, label='Hillary Clinton')
    plt.plot(pcarray.iloc[:, xval][labels == 1], pcarray.iloc[:, yval][labels == 1], 'ro',
             alpha=0.6, label='Donald Trump')
    plt.xlabel('PC{}'.format(xval + 1))
    plt.ylabel('PC{}'.format(yval + 1))
    plt.legend(loc='best', numpoints=1)
    plt.show()


def make_biplot(pcscores, labels, loadings, xval=0, yval=1, max_arrow=0.2):
    plt.figure()
    n = loadings.shape[1]
    scalex = 1.0 / (pcscores.iloc[:, xval].max() - pcscores.iloc[:, xval].min())  # Rescaling to be from -1 to +1
    scaley = 1.0 / (pcscores.iloc[:, yval].max() - pcscores.iloc[:, yval].min())

    plt.plot(pcscores.iloc[:, xval][labels == 0] * scalex, pcscores.iloc[:, yval][labels == 0] * scaley,
             'bo', alpha=0.6, label='Hillary Clinton')
    plt.plot(pcscores.iloc[:, xval][labels == 1] * scalex, pcscores.iloc[:, yval][labels == 1] * scaley,
             'ro', alpha=0.6, label='Donald Trump')

    for i in range(n):
        # Only plot the longer ones
        length = sqrt(loadings.iloc[xval, i]**2 + loadings.iloc[yval, i]**2)
        if length < max_arrow:
            continue

        plt.arrow(0, 0, loadings.iloc[xval, i], loadings.iloc[yval, i], color='g', alpha=0.5)
        plt.text(loadings.iloc[xval, i] * 1.15, loadings.iloc[yval, i] * 1.15,
                 loadings.columns.tolist()[i], color='k', ha='center', va='center')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('PC{}'.format(xval+1))
    plt.ylabel('PC{}'.format(yval+1))
    plt.legend(loc='best', numpoints=1)
    plt.grid()
    plt.show()
