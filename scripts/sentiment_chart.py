# Create a comparison of Clinton and Trump tweets based on sentiments

from analysis import Analyzer
from tweetloader import TweetLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load most recent tweets from Hillary Clinton and Donald Trump
# h = TweetLoader('HillaryClinton')
# t = TweetLoader('realDonaldTrump')
h = TweetLoader('', path='data/backup/', filename='hillary_2016-07-13.json')
t = TweetLoader('', path='data/backup/', filename='trump_2016-07-13.json')
h.load()
t.load()


# Assign label (second array) for Hillary(0)/Trump(1) tweets
label_array = np.array([0]*len(h.tweets) + [1]*len(t.tweets))

df_tweets = pd.concat([h.tweets['text'], t.tweets['text']], axis=0, join='outer', join_axes=None,
                      ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)

# Using the Analyzer class to get sentiments
mod = Analyzer(df_tweets, label_array)
mod.get_sentiment()

# Group together tweets, labels, and sentiments
temp = pd.concat([h.tweets, t.tweets], axis=0, join='outer', join_axes=None, ignore_index=True, levels=None)
df = pd.concat([temp, mod.sentiment, pd.DataFrame({'label': label_array})], axis=1, levels=None)


# Get Tweet text and URLs for embedding: https://twitter.com/{user}/status/{id}
def print_and_get_url(tweet):
    print tweet['text'].values[0]
    print 'https://twitter.com/{}/status/{}'.format(tweet['user.screen_name'].values[0], tweet['id'].values[0])

# Most positive and negative tweet
print_and_get_url(df.sort_values(by='positive', ascending=False)[df['label'] == 0])
print_and_get_url(df.sort_values(by='positive', ascending=False)[df['label'] == 1])
print_and_get_url(df.sort_values(by='negative', ascending=False)[df['label'] == 0])
print_and_get_url(df.sort_values(by='negative', ascending=False)[df['label'] == 1])

print_and_get_url(df.sort_values(by='fear', ascending=False)[df['label'] == 0])
print_and_get_url(df.sort_values(by='disgust', ascending=False)[df['label'] == 1])


# Tally up the results
results = df.groupby(by='label', sort=False).sum()
normalized = df.groupby(by='label', sort=False).sum()
normalized.loc[0] = normalized.loc[0]/len(h.tweets)
normalized.loc[1] = normalized.loc[1]/len(t.tweets)
results.index = ['Clinton', 'Trump']
normalized.index = ['Clinton', 'Trump']
# Drop extra columns
results.drop(['id', 'index', 'level_0'], axis=1, inplace=True)
normalized.drop(['id', 'index', 'level_0'], axis=1, inplace=True)

# Chart the results
ind = np.arange(10)
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(ind, results.loc['Clinton'], width, color='blue', alpha=0.6)
rects2 = ax.bar(ind + width, results.loc['Trump'], width, color='red', alpha=0.6)
ax.set_ylabel('Word Count')
ax.set_xticks(ind + width)
ax.set_xticklabels(results.columns.tolist())
plt.xticks(rotation='vertical')
plt.tight_layout()
ax.legend((rects1[0], rects2[0]), ('Hillary Clinton', 'Donald Trump'), loc='best')

# Normalized
fig, ax = plt.subplots()
rects1 = ax.bar(ind, normalized.loc['Clinton'], width, color='blue', alpha=0.6)
rects2 = ax.bar(ind + width, normalized.loc['Trump'], width, color='red', alpha=0.6)
ax.set_ylabel('Word Count / Number of Tweets')
ax.set_xticks(ind + width)
ax.set_xticklabels(results.columns.tolist())
plt.xticks(rotation='vertical')
plt.tight_layout()
ax.legend((rects1[0], rects2[0]), ('Hillary Clinton', 'Donald Trump'), loc='best')
plt.savefig('figures/sentiment_normalized.png')

# Positivity distribution
df['positivity'] = df['positive'] - df['negative']
bins = [s-5 for s in range(11)]
plt.hist(df[df['label'] == 0]['positivity'], bins=bins, color='blue', alpha=0.6, label='Hillary Clinton', normed=True)
plt.hist(df[df['label'] == 1]['positivity'], bins=bins, color='red', alpha=0.6, label='Donald Trump', normed=True)
plt.legend()
plt.xlabel('Positivity')
plt.ylabel('Fraction of Tweets')
plt.savefig('figures/positivity_distribution.png')
