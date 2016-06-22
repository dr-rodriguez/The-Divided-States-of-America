# Create word cloud based on candidates tweets
from tweetloader import TweetLoader
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread
import matplotlib.pyplot as plt


# Cloud function
def make_cloud(text, image, size=10, filename='figures/cloud.png', max_words=200, horizontal=0.8):
    words = ' '.join(text)

    # Remove URLs, 'RT' text, screen names, etc
    my_stopwords = ['RT', 'amp', 'lt']
    words_no_urls = ' '.join([word for word in words.split()
                              if 'http' not in word and word not in my_stopwords and not word.startswith('@')
                              ])

    # Add stopwords, if needed
    stopwords = STOPWORDS.copy()
    stopwords.add("RT")
    stopwords.add('amp')
    stopwords.add('lt')

    # Load up a logo as a mask & color image
    logo = imread(image)

    # Generate colors
    image_colors = ImageColorGenerator(logo)

    # Generate plot
    wc = WordCloud(stopwords=stopwords, mask=logo, color_func=image_colors, scale=0.8,
                   max_words=max_words, background_color='white', random_state=42, prefer_horizontal=horizontal)

    wc.generate(words_no_urls)

    plt.figure(figsize=(size, size))
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(filename)

# Load tweets
h = TweetLoader('HillaryClinton', track_location=False)
h.load()
t = TweetLoader('realDonaldTrump', track_location=False)
t.load()

# Hillary cloud
make_cloud(h.tweets['text'], 'logos/DemocraticLogo2.png', 10, 'figures/clinton_cloud2.png')

# Trump cloud
make_cloud(t.tweets['text'], 'logos/RepublicanLogo2.png', 10, 'figures/trump_cloud2.png')

