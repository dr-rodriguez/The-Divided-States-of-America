# A file to store helper functions

import requests
import simplejson
from requests_oauthlib import OAuth1 # used for twitter's api
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import numpy as np
from bokeh.plotting import figure

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

    if labels is not None:
        plt.plot(pcscores.iloc[:, xval][labels == 0] * scalex, pcscores.iloc[:, yval][labels == 0] * scaley,
                 'bo', alpha=0.6, label='Hillary Clinton')
        plt.plot(pcscores.iloc[:, xval][labels == 1] * scalex, pcscores.iloc[:, yval][labels == 1] * scaley,
                 'ro', alpha=0.6, label='Donald Trump')
    else:
        plt.plot(pcscores.iloc[:, xval] * scalex, pcscores.iloc[:, yval] * scaley,
                 'bo', alpha=0.6)

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
    if labels is not None: plt.legend(loc='best', numpoints=1)
    plt.grid()
    plt.show()


def top_factors(data, comp, count=10, ascend=False):
    x = data.iloc[:, comp].copy()
    x.sort_values(ascending=ascend, inplace=True)
    print(x[:count])
    return


def pretty_cm(cm, label_names=['Hillary', 'Trump'], show_sum=True):
    table = pd.DataFrame(cm, columns=['P-'+s for s in label_names], index=['T-'+s for s in label_names])
    print(table)
    if show_sum:
        print('Sum of columns: {}'.format(cm.sum(axis=0)))
        print('Sum of rows: {}'.format(cm.sum(axis=1)))


def get_tweets_label(df, label, size=10, colname='predict'):
    temp_df = df[colname] == label
    return df[temp_df].sample(size)

# Generate color bar
# Adapted From: http://stackoverflow.com/questions/32614953/can-i-plot-a-colorbar-for-a-bokeh-heatmap
def generate_colorbar(palette, low=0, high=1, plot_height=700, plot_width=1100, orientation='h'):

    y = np.linspace(low, high, len(palette))
    dy = y[1]-y[0]
    if orientation.lower() == 'v':
        fig = figure(tools="", x_range=[0, 1], y_range=[low, high], plot_width = plot_width, plot_height=plot_height)
        fig.toolbar_location = None
        fig.xaxis.visible = None
        fig.rect(x=0.5, y=y, color=palette, width=1, height=dy)
    elif orientation.lower() == 'h':
        fig = figure(tools="", y_range=[0, 1], x_range=[low, high],plot_width = plot_width, plot_height=plot_height)
        fig.toolbar_location = None
        fig.yaxis.visible = None
        fig.rect(x=y, y=0.5, color=palette, width=dy, height=1)
    return fig


