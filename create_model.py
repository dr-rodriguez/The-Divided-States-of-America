# Script to carry out my model processing

from helper_functions import print_dtm, make_plot, make_biplot, top_factors
import pandas as pd
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import PCA
import numpy as np
from tweetloader import TweetLoader
import matplotlib.pyplot as plt

# Some global defaults
max_tweets = 500
max_words = 150
load_tweets = True
get_new_tweets = False
save_tweets = False

# Load most recent tweets from Hillary Clinton and Donald Trump
h = TweetLoader('HillaryClinton')
t = TweetLoader('realDonaldTrump')

if load_tweets:
    h.load()
    t.load()

if get_new_tweets:
    h.timeline(max_tweets, exclude_replies='true', include_rts='false')
    t.timeline(max_tweets, exclude_replies='true', include_rts='false')

if save_tweets:
    h.save()
    t.save()

# Merge tweets together, get most common terms
df_tweets = pd.concat([h.tweets['text'], t.tweets['text']], axis=0, join='outer', join_axes=None,
                      ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)
str_list = ' '.join([tweet for tweet in df_tweets])

porter = PorterStemmer()  # use stemming
stop_words = set(stopwords.words('english'))  # remove stopwords
stop_words.update([s for s in string.punctuation] +
                  [u'\u2014', u'\u2019', u'\u201c', u'\xf3', u'\u201d', u'\u2014@', u'://', u'!"', u'"@',
                   u'."', u'.@', u'co'])
# Political terms and Twitter handles to remove
stop_words.update(['hillary', 'clinton', 'donald', 'trump', 'clinton2016',
                   'trump2016', 'hillary2016', 'makeamericagreatagain'])
stop_words.update(['realdonaldtrump', 'hillaryclinton'])

words = Counter([porter.stem(i.lower()) for i in wordpunct_tokenize(str_list)
                 if i.lower() not in stop_words and not i.lower().startswith('http')])
top_words = dict(words.most_common(max_words))

# Check each tweet against the most common terms
dtm = []
for tweet in df_tweets:

    # Make empty row
    newrow = dict()
    for term in top_words.keys():
        newrow[term] = 0

    tweetwords = [porter.stem(i.lower()) for i in wordpunct_tokenize(tweet)
             if i.lower() not in stop_words and not i.lower().startswith('http')]

    for word in tweetwords:
        if word in top_words.keys():
            newrow[word] += 1

    dtm.append(newrow)

# Quickly look at some results
print_dtm(dtm, df_tweets, 45)

# Word chart
top_25 = dict(words.most_common(25))
fig, ax = plt.subplots()
ind = np.arange(len(top_25))
width = 0.35
ax.bar(ind + width, top_25.values(), width, color='b')
ax.set_ylabel('Word Count')
ax.set_xticks(ind + width)
plt.xticks(ind + width, top_25.keys(), rotation='vertical')
plt.tight_layout()

# Assign label (second array) for Hillary/Trump tweets
label_array = np.array([0]*len(h.tweets) + [1]*len(t.tweets))

# Pass common terms to PCA, retaining components that describe ~70% of the variance
df_dtm = pd.DataFrame(dtm, columns=top_words.keys())
pca = PCA(n_components=0.7)
pca.fit(df_dtm)
pcscores = pd.DataFrame(pca.transform(df_dtm))
pcscores.columns = ['PC'+str(i+1) for i in range(pcscores.shape[1])]
loadings = pd.DataFrame(pca.components_, columns=top_words.keys())
load_squared = loadings.transpose()**2
load_squared.columns = ['PC'+str(i+1) for i in range(pcscores.shape[1])]

# Exploratory plots
make_plot(pcscores, label_array, 0, 1)
make_biplot(pcscores, label_array, loadings, 2, 3)

# Top terms in components
top_factors(load_squared, 0)
top_factors(load_squared, 1)

# Create randomized index, split 80-20 into training/test sets
ind = np.arange(len(df_tweets))
np.random.shuffle(ind)

# Shuffle arrays
pcscores = pcscores.iloc[ind]
label_array = label_array[ind]

cut = int(len(ind)*0.8)
df_train = pcscores.iloc[:cut]
df_test = pcscores.iloc[cut:]
train_label = label_array[:cut]
test_label = label_array[cut:]

# TODO: Run SCV on training set

# TODO: Compare to test set

# TODO: Figure out how to get confusion matrix

