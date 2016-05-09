# Script to carry out my model processing

from helper_functions import grab_user_tweets

# Load most recent 2000 tweets from Hillary Clinton and Donald Trump
df_hillary = grab_user_tweets('HillaryClinton', 2000)
df_trump = grab_user_tweets('realDonaldTrump', 2000)

# TODO: Merge tweets together, get 100-200 most common terms

# TODO: Pass common terms to PCA, retaining components that describe ~70% of the variance

# TODO: Assign label (second array) for Hillary/Trump tweets

# TODO: Create randomized index, split 80-20 into training/test sets

# TODO: Run SCV on training set

# TODO: Compare to test set

# TODO: Figure out how to get confusion matrix

