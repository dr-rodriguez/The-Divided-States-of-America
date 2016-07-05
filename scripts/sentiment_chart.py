# Create a comparison of Clinton and Trump tweets based on sentiments

from analysis import Analyzer
from tweetloader import TweetLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load most recent tweets from Hillary Clinton and Donald Trump
h = TweetLoader('HillaryClinton')
t = TweetLoader('realDonaldTrump')

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
df = pd.concat([df_tweets, mod.sentiment, pd.DataFrame({'label': label_array})], axis=1)

# Most positive and negative tweet
print df.sort_values(by='positive', ascending=False)[df['label'] == 0]['text'].values[0]
print df.sort_values(by='positive', ascending=False)[df['label'] == 1]['text'].values[0]

print df.sort_values(by='negative', ascending=False)[df['label'] == 0]['text'].values[0]
print df.sort_values(by='negative', ascending=False)[df['label'] == 1]['text'].values[0]

# Tally up the results
results = df.groupby(by='label', sort=False).sum()
normalized = df.groupby(by='label', sort=False).sum()
normalized.loc[0] = normalized.loc[0]/len(h.tweets)
normalized.loc[1] = normalized.loc[1]/len(t.tweets)
results.index = ['Clinton', 'Trump']
normalized.index = ['Clinton', 'Trump']

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
plt.hist(df[df['label'] == 0]['positivity'], bins=bins, color='blue', alpha=0.6, label='Hillary Clinton')
plt.hist(df[df['label'] == 1]['positivity'], bins=bins, color='red', alpha=0.6, label='Donald Trump')
plt.legend()
plt.xlabel('Positivity')
plt.ylabel('Number of Tweets')
plt.savefig('figures/positivity_distribution.png')
