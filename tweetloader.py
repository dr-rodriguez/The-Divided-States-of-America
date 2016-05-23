# Class for loading and processing tweets
import requests
import simplejson
from requests_oauthlib import OAuth1 # used for twitter's api
from pandas.io.json import json_normalize
import pandas as pd
import time
import os

class TweetLoader:
    def __init__(self, screen_name, filename='tweets.json'):
        self.screen_name = screen_name
        self.tweets = []

        if screen_name == 'HillaryClinton':
            self.filename = 'hillary.json'
        elif screen_name == 'realDonaldTrump':
            self.filename = 'trump.json'
        else:
            self.filename = filename

        print('Using {}'.format(self.filename))

        # Twitter authorization
        with open("twitter_secrets.json.nogit") as f:
            secrets = simplejson.loads(f.read())

        self.auth = OAuth1(
            secrets["api_key"],
            secrets["api_secret"],
            secrets["access_token"],
            secrets["access_token_secret"]
        )

        return

    def search(self, max_tweets=200, exclude_replies='true', include_rts='false'):
        """
            Load Twitter timeline for the specified user

            :param screen_name: Twitter screen name to process
            :param max_tweets: maximum number of tweets to get. (Default: 3200, the API maximum)
            :param exclude_replies: exclude replies? (Default: true)
            :param include_rts: include retweets? (Default: false)
            :return: pandas DataFrame of tweets
            """

        # API only allows up to 200 tweets per page

        max_pages = int(min(max_tweets, 3200) // 200)
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
                params = {'screen_name': self.screen_name, 'count': count, 'lang': 'en',
                          'exclude_replies': exclude_replies, 'include_rts': include_rts}
            else:
                max_id = data[-1]['id'] - 1
                params = {'screen_name': self.screen_name, 'count': count, 'lang': 'en',
                          'exclude_replies': exclude_replies, 'include_rts': include_rts,
                          'max_id': max_id}

            r = requests.get(url, auth=self.auth, params=params)
            data = simplejson.loads(r.text)

            if page == 0:
                df = json_normalize(data)
            else:
                df = df.append(json_normalize(data), ignore_index=True)

            page += 1

        if len(self.tweets) == 0:
            self.tweets = df[['id', 'text']]
        else:
            self.tweets = self.merge(df[['id', 'text']])

        return

    def load(self):
        if not os.path.isfile('data/' + self.filename):
            print('File does not exist. Create it first.')
            return

        data = pd.read_json('data/' + self.filename)

        if len(self.tweets) == 0:
            self.tweets = data
        else:
            self.tweets = self.merge(data)

        return

    def merge(self, data):
        # Merge data
        newdf = pd.concat([self.tweets, data], axis=0, join='outer', join_axes=None,
                          ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)

        # Eliminate duplicates
        newdf = newdf.drop_duplicates(subset=['id'])

        return newdf

    def save(self):
        self.tweets[['id', 'text']].to_json('data/'+self.filename)  # only save ID and the tweet text to save space
        return

    def makebackup(self):
        data = pd.read_json('data/' + self.filename)
        newfile = self.filename[:-5] + '_' + time.strftime("%Y-%m-%d") + '.json'
        data.to_json('data/backup/' + newfile)
        return
