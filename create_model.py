# Script to read in tweets and process them

import requests
import simplejson
from requests_oauthlib import OAuth1 # used for twitter's api
from pandas.io.json import json_normalize

with open("twitter_secrets.json.nogit") as f:
    secrets = simplejson.loads(f.read())

# OAuth1 is a module in requests_oauthlib
auth = OAuth1(
    secrets["api_key"],
    secrets["api_secret"],
    secrets["access_token"],
    secrets["access_token_secret"]
)

# TODO: Grab most recent ~2000 user tweets for Hillary and Trump
# Load user data
page = 0
url = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
screen_name = 'strakul'
for i in range(2):
    if page == 0:
        params = {'screen_name': screen_name, 'count': 10, 'exclude_replies': 'true', 'include_rts': 'false'}
    else:
        max_id = data[-1]['id'] - 1
        params = {'screen_name': screen_name, 'count': 10, 'exclude_replies': 'true', 'include_rts': 'false',
                  'max_id': max_id}

    r = requests.get(url, auth=auth, params=params)
    data = simplejson.loads(r.text)

    if page == 0:
        df = json_normalize(data)
    else:
        df = df.append(json_normalize(data), ignore_index=True)

    page += 1

# TODO: Merge tweets together, get 100-200 most common terms

# TODO: Pass common terms to PCA, retaining components that describe ~70% of the variance

# TODO: Assign label (second array) for Hillary/Trump tweets

# TODO: Create randomized index, split 80-20 into training/test sets

# TODO: Run SCV on training set

# TODO: Compare to test set

# TODO: Figure out how to get confusion matrix

