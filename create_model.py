# Script to carry out model processing

from analysis import Analyzer
from tweetloader import TweetLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Some global defaults
max_words = 200

# Load most recent tweets from Hillary Clinton and Donald Trump
h = TweetLoader('HillaryClinton')
t = TweetLoader('realDonaldTrump')

h.load()
t.load()

# Assign label (second array) for Hillary(0)/Trump(1) tweets
label_array = np.array([0]*len(h.tweets) + [1]*len(t.tweets))

# Merge tweets together, pass to Analyzer
df_tweets = pd.concat([h.tweets['text'], t.tweets['text']], axis=0, join='outer', join_axes=None,
                      ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)

# Using the Analyzer class
mod = Analyzer(df_tweets, label_array, max_words=max_words, load_pca=False, load_svm=False, use_sentiment=True)

# mod.get_words()
# mod.create_dtm()
# mod.run_pca()
# mod.get_sentiment()
# test_predict, test_label = mod.run_svm()

# One-line alternative with defaults
test_predict, test_label = mod.create_full_model()

# Check a PCA plot
mod.make_biplot(2, 3, 0.2)

# Check results
cm = mod.make_confusion_matrix(test_label, test_predict, normalize=False, axis=0, label_names=['Hillary', 'Trump'])

# Examine sentiments
df = pd.concat([df_tweets, mod.sentiment, pd.DataFrame({'label': label_array})], axis=1)

bins = [s-5 for s in range(11)]
plt.hist(df[df['label'] == 0]['positivity'], bins=bins, color='blue', alpha=0.6, label='Hillary Clinton')
plt.hist(df[df['label'] == 1]['positivity'], bins=bins, color='red', alpha=0.6, label='Donald Trump')
plt.legend()
plt.xlabel('Positivity')
plt.ylabel('Number of Tweets')
plt.savefig('figures/positivity_distribution.png')

# Save model results
os.system('rm model/*')  # Clear the prior models first
mod.save_words()
mod.save_pca()
mod.save_svm()

