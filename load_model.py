# Load the model previously created
from analysis import Analyzer
from tweetloader import TweetLoader
import matplotlib.pyplot as plt
import pandas as pd

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

# Make a plot
df = s.tweets['geo.coordinates']
bad = df.apply(lambda x: x is None)
df = df[~bad]  # Eliminate missing values

lat = df.apply(lambda x: x[0])
lon = df.apply(lambda x: x[1])

labels = pd.Series(predict)[~bad]
plt.scatter(lon[labels==0], lat[labels==0], color='b', alpha=0.6)
plt.scatter(lon[labels==1], lat[labels==1], color='r', alpha=0.6)

from mpl_toolkits.basemap import Basemap
# US_BOUNDING_BOX = "-125.00,24.94,-66.93,49.59"
m = Basemap(projection='gall', llcrnrlat=18, urcrnrlat=55, llcrnrlon=-126, urcrnrlon=-65,
            resolution='l', area_thresh=10000)

# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates(color='grey')
m.drawcountries(linewidth=1.5)
m.drawmapboundary(fill_color='steelblue')
m.fillcontinents(color='gainsboro', lake_color='steelblue')

x, y = m(lon.values, lat.values)

m.plot(x, y, 'o', color='g', alpha=0.8)
plt.savefig('figures/map.png')
