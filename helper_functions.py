# A file to store helper functions

import requests
import simplejson
from requests_oauthlib import OAuth1 # used for twitter's api
from pandas.io.json import json_normalize


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
            params = {'screen_name': screen_name, 'count': count,
                      'exclude_replies': exclude_replies, 'include_rts': include_rts}
        else:
            max_id = data[-1]['id'] - 1
            params = {'screen_name': screen_name, 'count': count,
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