# Script to carry out my model processing

from helper_functions import grab_user_tweets, print_dtm, make_plot
import pandas as pd
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import PCA
import numpy as np

# Load most recent 2000 tweets from Hillary Clinton and Donald Trump
df_hillary = grab_user_tweets('HillaryClinton', 200)
df_trump = grab_user_tweets('realDonaldTrump', 200)

# Merge tweets together, get most common terms
df_tweets = pd.concat([df_hillary['text'], df_trump['text']], axis=0, join='outer', join_axes=None,
                      ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)
str_list = ' '.join([tweet for tweet in df_tweets])

porter = PorterStemmer()  # use stemming
stop_words = set(stopwords.words('english'))  # remove stopwords
stop_words.update([s for s in string.punctuation] +
                  [u'\u2014', u'\u2019', u'\u201c', u'\xf3', u'\u201d', u'\u2014@', u'://', u'!"', u'"@',
                   u'."', u'.@', u'co'])

words = Counter([porter.stem(i.lower()) for i in wordpunct_tokenize(str_list)
                 if i.lower() not in stop_words and not i.lower().startswith('http')])
top_words = dict(words.most_common(100))

# Check each tweet against the most common terms
dtm = []
for tweet in df_tweets:

    # Make empty row
    newrow = dict()
    for term in top_words.keys():
        newrow[term] = 0

    words = [porter.stem(i.lower()) for i in wordpunct_tokenize(tweet)
             if i.lower() not in stop_words and not i.lower().startswith('http')]

    for word in words:
        if word in top_words.keys():
            newrow[word] += 1

    dtm.append(newrow)

# Quickly look at some results
print_dtm(dtm, df_tweets, 42)

# Assign label (second array) for Hillary/Trump tweets
label_array = np.array([0]*len(df_hillary) + [1]*len(df_trump))

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
make_biplot(pcscores, label_array, loadings, 0, 1)

# TODO: Create randomized index, split 80-20 into training/test sets

# TODO: Run SCV on training set

# TODO: Compare to test set

# TODO: Figure out how to get confusion matrix

