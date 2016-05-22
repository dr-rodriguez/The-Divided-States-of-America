# Script to carry out my model processing

from helper_functions import grab_user_tweets, print_dtm
import pandas as pd
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer

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

# TODO: Pass common terms to PCA, retaining components that describe ~70% of the variance

# TODO: Assign label (second array) for Hillary/Trump tweets

# TODO: Create randomized index, split 80-20 into training/test sets

# TODO: Run SCV on training set

# TODO: Compare to test set

# TODO: Figure out how to get confusion matrix

