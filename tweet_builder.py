#!/usr/bin/env python

# Script meant to be run once per day to populate the tweet database
from tweetloader import TweetLoader
import time

max_tweets = 1000

h = TweetLoader('HillaryClinton', track_location=False)
h.load()
h.timeline(max_tweets, exclude_replies='true', include_rts='false')
h.save()

t = TweetLoader('realDonaldTrump', track_location=False)
t.load()
t.timeline(max_tweets, exclude_replies='true', include_rts='false')
t.save()

bs = TweetLoader('BernieSanders', filename='sanders.json', track_location=False)
bs.load()
bs.timeline(max_tweets, exclude_replies='true', include_rts='false')
bs.save()

# Search results
s = TweetLoader(filename='search.json', track_location=True)
s.load()
query = 'politic OR trump OR hillary OR clinton OR election'
s.search(query, max_tweets)
s.save()

# For fun, gather CS19 tweets
s2 = TweetLoader(filename='coolstars.json', track_location=False, path='coolstars19/data/')
s2.load()
query = '#CS19'
s2.search(query, 2000, hard_remove=False, remove_rts=False)
s2.save()

print(time.strftime("%Y-%m-%d"))
print('{} tweets gathered for Hillary Clinton'.format(len(h.tweets)))
print('{} tweets gathered for Donald Trump'.format(len(t.tweets)))
print('{} tweets gathered by search terms'.format(len(s.tweets)))

