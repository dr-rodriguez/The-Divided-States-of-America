# Create a Bokeh map of the tweets

# Load the model previously created
from analysis import Analyzer
from tweetloader import TweetLoader
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.us_states import data as states

# Some global defaults
max_words = 150

# Load most recent tweets from Hillary Clinton and Donald Trump
s = TweetLoader(filename='search.json', track_location=True)
s.load()

# Calculate and grab model results
mod = Analyzer(s.tweets['text'], max_words=max_words, load_pca=True, load_svm=True)
predict = mod.load_full_model()

# Make a Bokeh plot
df = s.tweets['geo.coordinates']
bad = df.apply(lambda x: x is None)
df = df[~bad]  # Eliminate missing values

lat = df.apply(lambda x: x[0])
lon = df.apply(lambda x: x[1])

# Remove Alaska and Hawaii
del states["HI"]
del states["AK"]

state_xs = [states[code]["lons"] for code in states]
state_ys = [states[code]["lats"] for code in states]

p = figure(title="Twitter Results", toolbar_location="left", plot_width=1100, plot_height=700)
p.patches(state_xs, state_ys, fill_alpha=0.0, line_color="black", line_width=2, line_alpha=0.3)

p.scatter(lon, lat)

output_file("figures/bokeh_map.html")

show(p)

# TODO: See if tweets can be matched by patch (ie, state)

# TODO: If tweets matched by state: color-code each state via model prediction
# Remember to eliminate bad entries
