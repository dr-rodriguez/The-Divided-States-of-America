# Script to carry out my model processing

from helper_functions import print_dtm, make_plot, make_biplot, top_factors, pretty_cm
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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn import svm, grid_search

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

# Remove stop words
stop_words = set(stopwords.words('english'))
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


# Pass common terms to PCA, retaining components that describe ~70% of the variance
df_dtm = pd.DataFrame(dtm, columns=top_words.keys())
pca = PCA(n_components=0.7)
pca.fit(df_dtm)
pcscores = pd.DataFrame(pca.transform(df_dtm))
pcscores.columns = ['PC'+str(i+1) for i in range(pcscores.shape[1])]
loadings = pd.DataFrame(pca.components_, columns=top_words.keys())
load_squared = loadings.transpose()**2
load_squared.columns = ['PC'+str(i+1) for i in range(pcscores.shape[1])]

# Assign label (second array) for Hillary(0)/Trump(1) tweets
label_array = np.array([0]*len(h.tweets) + [1]*len(t.tweets))

# Exploratory plots
make_plot(pcscores, label_array, 0, 1)
make_biplot(pcscores, label_array, loadings, 2, 3)

# Top terms in components
top_factors(load_squared, 0)
top_factors(load_squared, 2)

# Create randomized index, split 80-20 into training/test sets
df_train, df_test, train_label, test_label = train_test_split(pcscores, label_array,
                                                              test_size=0.2, random_state=42)

# Run SVC (support vector classifier) on training set
# with cross validation to tune parameters
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters, cv=5, error_score=0)
clf.fit(df_train, train_label)
print('Best parameters: {}'.format(clf.best_params_))
test_predict = clf.predict(df_test)

# Examine confusion matrix
cm = confusion_matrix(test_label, test_predict)
cm_normalized = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]  # axis=0 by column(precision), axis=1 by row(recall)
label_names=['Hillary', 'Trump']
pretty_cm(cm, label_names)
pretty_cm(cm_normalized, label_names, show_sum=False)
print(classification_report(test_label, test_predict, target_names=label_names))
"""
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
The recall is intuitively the ability of the classifier to find all the positive samples.
These quantities are also related to the (F_1) score, which is defined as the harmonic mean of precision and recall.
F1 = 2\frac{P \times R}{P+R}
"""

# Checking false positives
ind = np.where(test_label != test_predict)
orig_ind = df_test.iloc[ind].index
# df_tweets[orig_ind] # All
df_tweets[ orig_ind[ label_array[orig_ind]==0 ] ]  # Hillary only
df_tweets[ orig_ind[ label_array[orig_ind]==1 ] ]  # Trump only

# Save model
joblib.dump(clf, 'model/model.pkl')
# clf = joblib.load('model/model.pkl')