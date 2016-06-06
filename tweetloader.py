# Class for loading and processing tweets
import requests
import simplejson
from requests_oauthlib import OAuth1 # used for twitter's api
from pandas.io.json import json_normalize
import pandas as pd
import time
import os
import numpy as np


class TweetLoader:

    def __init__(self, screen_name='', filename='tweets.json', track_location=False, path='data/'):
        self.screen_name = screen_name
        self.tweets = []
        self.columns = ['id', 'text', 'created_at', 'user.screen_name']  # which information to save
        self.track_location = track_location
        self.verbose = False
        self.path = path

        # Save location information
        if track_location:
            self.columns = self.columns + ['geo.coordinates', 'user.location']

        if screen_name == 'HillaryClinton':
            self.filename = 'hillary.json'
        elif screen_name == 'realDonaldTrump':
            self.filename = 'trump.json'
        elif screen_name == 'BernieSanders':
            self.filename = 'sanders.json'
        else:
            self.filename = filename

        if self.verbose: print('Using {}'.format(self.filename))

        # Twitter authorization
        with open("twitter_secrets.json.nogit") as f:
            secrets = simplejson.loads(f.read())

        self.auth = OAuth1(
            secrets["api_key"],
            secrets["api_secret"],
            secrets["access_token"],
            secrets["access_token_secret"]
        )

    def timeline(self, max_tweets=200, exclude_replies='true', include_rts='false'):
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
            self.tweets = df[self.columns]
        else:
            self.tweets = self.merge(df[self.columns])

    def stream(self):
        # Not yet implemented

        # Accessing the stream API (live tweets)
        US_BOUNDING_BOX = "-125.00,24.94,-66.93,49.59"

    def search(self, query, max_tweets=200, remove_rts=True, hard_remove=True):

        # Search API only allows 100 tweets per page
        max_pages = int(max_tweets) // 100
        if max_pages < 1:
            max_pages = 1

        if max_tweets < 100:
            count = int(max_tweets)
        else:
            count = 100

        # Prepare query
        if remove_rts:
            query += ' -filter:retweets'
        if hard_remove:
            query += ' -RT'  # eliminates anything with RT, which may not always be a retweet

        # encoded_query = urllib.quote_plus(query)

        page = 0
        url = 'https://api.twitter.com/1.1/search/tweets.json'
        for i in range(max_pages):
            if page == 0:
                params = {'q': query, 'result_type': 'recent', 'count': count, 'lang': 'en'}
            else:
                max_id = data[-1]['id'] - 1
                params = {'q': query, 'result_type': 'recent', 'count': count, 'lang': 'en', 'max_id': max_id}

            r = requests.get(url, auth=self.auth, params=params)
            data = simplejson.loads(r.text)['statuses']

            if len(data) == 0:
                if self.verbose: print('No more results found')
                break

            if page == 0:
                df = json_normalize(data)
            else:
                df = df.append(json_normalize(data))

            page += 1

        # Check that all columns are there, if not add empty ones
        for col in self.columns:
            if col not in df.columns:
                df[col] = pd.Series([np.nan] * len(df), index=df.index)

        if len(self.tweets) == 0:
            self.tweets = df[self.columns]
        else:
            self.tweets = self.merge(df[self.columns])

        # Filter by location
        if self.track_location:
            if self.verbose: print('Filtering by location')
            self.get_geo()

        return

    def get_geo(self):
        """
        Get latitude and longitude from the Google API and a user's location
        """

        # Eliminate empty locations
        self.tweets = self.tweets[self.tweets['user.location'] != u'']

        # Search locations
        self.tweets.index = range(len(self.tweets))
        # self.tweets.reset_index(drop=True, inplace=True)
        droplist = []
        for i in range(len(self.tweets)):
            geo = self.tweets.iloc[i]['geo.coordinates']
            loc = self.tweets.iloc[i]['user.location']

            if geo is not None:
                try:
                    if not check_US(*geo):
                        droplist.append(i)
                        if self.verbose: print 'Removing: ', i, geo, loc
                    continue
                except TypeError:
                    if self.verbose: print geo

            # Using Google API for geocoding
            time.sleep(0.2)  # avoid API limits
            status, geo = geo_api_search(loc)

            if status in ['ZERO_RESULTS']:
                droplist.append(i)
                continue

            if status in ['OK']:
                # Remove non-US tweets
                if not check_US(*geo):
                    droplist.append(i)
                    if self.verbose: print 'Removing: ', i, geo, loc
                else:
                    self.tweets.set_value(i, 'geo.coordinates', geo)  # Add to tweet
            else:
                if self.verbose: print(status)
                if self.verbose: print geo, loc

        self.tweets.drop(droplist, inplace=True)

        return

    def load(self):
        if not os.path.isfile(self.path + self.filename):
            print('File does not exist. Create it first.')
            return

        data = pd.read_json(self.path + self.filename)

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
        # self.tweets.reset_index(drop=True).to_json(self.path+self.filename)
        self.tweets.index = range(len(self.tweets))
        self.tweets.to_json(self.path + self.filename)
        return

    def makebackup(self):
        data = pd.read_json(self.path + self.filename)
        newfile = self.filename[:-5] + '_' + time.strftime("%Y-%m-%d") + '.json'
        data.to_json(self.path + 'backup/' + newfile)
        return


def check_US(lat, lon):
    # US_BOUNDING_BOX = "-125.00,24.94,-66.93,49.59"
    good = False

    if (lat >= 24.94) and (lat <= 49.59) and (lon >= -125) and (lon <= -66.93):
        good = True
    else:
        good = False

    return good


def geo_api_search(loc):
    """
    Use Google API to get location information

    :param loc: string to parse
    :return: status, latitude, longitude
    """

    url = 'http://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': loc.strip(), 'sensor': 'false'}

    r = requests.get(url, params=params)
    data = simplejson.loads(r.text)

    if data['status'] in ['OK']:
        lat = data['results'][0]['geometry']['location']['lat']
        lon = data['results'][0]['geometry']['location']['lng']
        geo = [lat, lon]  # lat first, then lon
    else:
        geo = None

    return data['status'], geo

