#!/usr/bin/env python

# Script to create a backup for the tweets
# Meant to be run once per week

from tweetloader import TweetLoader
import time

h = TweetLoader('HillaryClinton')
h.load()
h.makebackup()

t = TweetLoader('realDonaldTrump')
t.load()
t.makebackup()

s = TweetLoader(filename='search.json', track_location=True)
s.load()
s.makebackup()

print(time.strftime("%Y-%m-%d"))
print('{} tweets backed up for Hillary Clinton'.format(len(h.tweets)))
print('{} tweets backed up for Donald Trump'.format(len(t.tweets)))
print('{} tweets backed up for search results'.format(len(s.tweets)))
