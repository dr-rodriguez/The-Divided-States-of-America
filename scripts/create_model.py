# Script to carry out model processing

from analysis import Analyzer
from tweetloader import TweetLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Some global defaults
max_words = 200

# Load most recent tweets from Hillary Clinton and Donald Trump
# For reproducability, use the backup of July 13, 2016
# h = TweetLoader('HillaryClinton')
# t = TweetLoader('realDonaldTrump')
h = TweetLoader('', path='data/backup/', filename='hillary_2016-07-13.json')
t = TweetLoader('', path='data/backup/', filename='trump_2016-07-13.json')
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
# mod.make_biplot(2, 3, max_arrow=0.2)

# Check results
cm = mod.make_confusion_matrix(test_label, test_predict, normalize=False, axis=0, label_names=['Clinton', 'Trump'])

# Heatmap plot
sns.heatmap(cm, square=True, xticklabels=['Hillary Clinton', 'Donald Trump'], annot=True, fmt="d",
            yticklabels=['Hillary Clinton', 'Donald Trump'], cbar=True, cbar_kws={"orientation": "vertical"}, cmap="BuGn")\
    .set(xlabel="Predicted Class", ylabel="Actual Class")
plt.savefig('figures/confusion_matrix.png')

# Save model results
os.system('rm model/*')  # Clear the prior models first
mod.save_words()
mod.save_pca()
mod.save_svm()

