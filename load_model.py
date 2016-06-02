# Load the model previously created
from analysis import Analyzer
from tweetloader import TweetLoader

# Some global defaults
max_words = 150

# Load most recent tweets from Hillary Clinton and Donald Trump
s = TweetLoader(filename='search.json', track_location=True)
s.load()

mod = Analyzer(s.tweets['text'], max_words=max_words, load_pca=True, load_svm=True)

# mod.load_words()
# mod.create_dtm()
# mod.run_pca()
# mod.run_svm()

# One-line alternative with defaults
predict = mod.load_full_model()
