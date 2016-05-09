# Script to read in tweets and process them

import requests
from requests_oauthlib import OAuth1 # used for twitter's api

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
url = 'https://api.twitter.com/1.1/statuses/user_timeline.json'

# TODO: Merge tweets together, get 100-200 most common terms

# TODO: Pass common terms to PCA, retaining components that describe ~70% of the variance

# TODO: Assign label (second array) for Hillary/Trump tweets

# TODO: Create randomized index, split 80-20 into training/test sets

# TODO: Run SCV on training set

# TODO: Compare to test set

# TODO: Figure out how to get confusion matrix

