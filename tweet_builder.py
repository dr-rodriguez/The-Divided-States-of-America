#!/usr/bin/env python

# Script meant to be run once per day to populate the tweet database
from tweetloader import TweetLoader
import time

max_tweets = 1000

h = TweetLoader('HillaryClinton')
h.load()
h.search(max_tweets, exclude_replies='true', include_rts='false')
h.save()

t = TweetLoader('realDonaldTrump')
t.load()
t.search(max_tweets, exclude_replies='true', include_rts='false')
t.save()

print(time.strftime("%Y-%m-%d"))
print('{} tweets gathered for Hillary Clinton'.format(len(h.tweets)))
print('{} tweets gathered for Donald Trump'.format(len(t.tweets)))

