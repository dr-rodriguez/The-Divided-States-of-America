from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import string
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn import svm, grid_search


class Analyzer:
    """
    Class for carrying out the analysis and model creation/loading
    """

    def __init__(self, data, labels=None, max_words=150, load_pca=False, load_svm=False):
        self.data = data  # Data matrix
        self.labels = labels  # Label array

        self.max_words = max_words
        self.dtm = []
        self.top_words = dict()
        self.words = Counter()

        self.load_pca = load_pca  # Load or compute the PCA?
        self.pca = None
        self.pcscores = None
        self.load_squared = None

        self.load_svm = load_svm
        self.svc = None

        # Use stemming
        self.porter = PorterStemmer()

        # Set stop words
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update([s for s in string.punctuation] +
                               [u'\u2014', u'\u2019', u'\u201c', u'\xf3', u'\u201d', u'\u2014@', u'://', u'!"', u'"@',
                                u'."', u'.@', u'co'])

        # Political terms and Twitter handles to remove
        self.stop_words.update(['hillary', 'clinton', 'donald', 'trump', 'clinton2016',
                                'trump2016', 'hillary2016', 'makeamericagreatagain'])
        self.stop_words.update(['realdonaldtrump', 'hillaryclinton', 'berniesanders'])

    def create_full_model(self):
        self.get_words()
        self.create_dtm()
        self.run_pca()
        return self.run_svm()

    def load_full_model(self):
        self.load_words()
        self.create_dtm()
        self.run_pca()
        return self.run_svm()

    def get_words(self):
        str_list = ' '.join([tweet for tweet in self.data])

        self.words = Counter([self.porter.stem(i.lower()) for i in wordpunct_tokenize(str_list)
                         if i.lower() not in self.stop_words and not i.lower().startswith('http')])
        self.top_words = dict(self.words.most_common(self.max_words))

    def save_words(self, filename='words.pkl'):
        joblib.dump(self.top_words, 'model/'+filename)

    def load_words(self, filename='words.pkl'):
        self.top_words = joblib.load('model/'+filename)

    def create_dtm(self):
        dtm = []
        for tweet in self.data:

            # Make empty row
            newrow = dict()
            for term in self.top_words.keys():
                newrow[term] = 0

            tweetwords = [self.porter.stem(i.lower()) for i in wordpunct_tokenize(tweet)
                          if i.lower() not in self.stop_words and not i.lower().startswith('http')]

            for word in tweetwords:
                if word in self.top_words.keys():
                    newrow[word] += 1

            dtm.append(newrow)

        self.dtm = dtm

    def run_pca(self, filename='pca.pkl'):
        df_dtm = pd.DataFrame(self.dtm, columns=self.top_words.keys())

        # Load or run the PCA
        if self.load_pca:
            print('Loading model/{}'.format(filename))
            pca = joblib.load('model/'+filename)
        else:
            pca = PCA(n_components=0.8)
            pca.fit(df_dtm)

        pcscores = pd.DataFrame(pca.transform(df_dtm))
        pcscores.columns = ['PC' + str(i + 1) for i in range(pcscores.shape[1])]
        loadings = pd.DataFrame(pca.components_, columns=self.top_words.keys())
        load_squared = loadings.transpose() ** 2
        load_squared.columns = ['PC' + str(i + 1) for i in range(pcscores.shape[1])]

        self.pcscores = pcscores
        self.load_squared = load_squared

        # Prep for save, just in case
        self.pca = pca

    def save_pca(self, filename='pca.pkl'):
        joblib.dump(self.pca, 'model/' + filename)

    def run_svm(self, filename='svm.pkl'):
        if not self.load_svm:
            df_train, df_test, train_label, test_label = train_test_split(self.pcscores, self.labels,
                                                                          test_size=0.2, random_state=42)
            parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
            svr = svm.SVC()
            clf = grid_search.GridSearchCV(svr, parameters, cv=5, error_score=0)
            clf.fit(df_train, train_label)
            print('Best parameters: {}'.format(clf.best_params_))
            prediction = clf.predict(df_test)
            self.svc = clf
            return prediction, test_label
        else:
            print('Loading model/{}'.format(filename))
            clf = joblib.load('model/'+filename)
            prediction = clf.predict(self.pcscores)
            self.svc = clf
            return prediction

    def save_svm(self, filename='svm.pkl'):
        joblib.dump(self.svc, 'model/' + filename)

    def make_confusion_matrix(self, test_label, test_predict, normalize=False, axis=0, label_names=['Hillary', 'Trump']):
        cm = confusion_matrix(test_label, test_predict)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]

        pretty_cm(cm)
        print(classification_report(test_label, test_predict, target_names=label_names))

        return cm


def pretty_cm(cm, label_names=['Hillary', 'Trump'], show_sum=False):
    table = pd.DataFrame(cm, columns=['P-' + s for s in label_names], index=['T-' + s for s in label_names])
    print(table)
    if show_sum:
        print('Sum of columns: {}'.format(cm.sum(axis=0)))
        print('Sum of rows: {}'.format(cm.sum(axis=1)))
    print('')




