# Create a Bokeh map of the tweets

# Load the model previously created
from analysis import Analyzer
from tweetloader import TweetLoader
from bokeh.plotting import figure, show, output_file, vplot, hplot
from bokeh.sampledata.us_states import data as states
import reverse_geocoder as rg
import pandas as pd
import numpy as np
from helper_functions import generate_colorbar

# Some global defaults
max_words = 150

# Load most recent tweets from Hillary Clinton and Donald Trump
s = TweetLoader(filename='search.json', track_location=True)
s.load()

# Calculate and grab model results
mod = Analyzer(s.tweets['text'], max_words=max_words, load_pca=True, load_svm=True)
predict = mod.load_full_model()  # Hillary=0  Trump=1
s.tweets['predict'] = predict

# Clean up missing coordinates
df = s.tweets['geo.coordinates']
bad = df.apply(lambda x: x is None)
df = df[~bad]
s.tweets = s.tweets[~bad]

lat = df.apply(lambda x: x[0])
lon = df.apply(lambda x: x[1])
# lat, lon = zip(*df)  # Alternate

# Remove Alaska and Hawaii
del states["HI"]
del states["AK"]

# Match tweets by state
coordinates = zip(lat, lon)
results = rg.search(coordinates)  # default mode = 2
# print results
state_match = [results[row]['admin1'] for row,_ in enumerate(results)]
s.tweets['state'] = state_match
cc = pd.Series([results[row]['cc'] for row,_ in enumerate(results)])
# Check if not in US:
good = cc == 'US'
if len(cc)>0:
    s.tweets.index = range(len(s.tweets))
    lon.index = range(len(lon))
    lat.index = range(len(lat))
    s.tweets = s.tweets[good]
    lon = lon[good]
    lat = lat[good]

# Make first plot
state_xs = [states[code]["lons"] for code in states]
state_ys = [states[code]["lats"] for code in states]

p = figure(title="Twitter Results", toolbar_location="left", plot_width=1100, plot_height=700)
p.patches(state_xs, state_ys, fill_alpha=0.0, line_color="black", line_width=2, line_alpha=0.3)

p.scatter(lon, lat)

output_file("figures/bokeh_map.html")

show(p)

# Group by state and get average prediction value
df_predict = s.tweets[['state', 'predict']].groupby(by='state').mean()

# Rename 'Washington, D.C.' to 'District of Columbia' to match states list
state_names = df_predict.index.tolist()
ind = state_names.index('Washington, D.C.')
state_names[ind] = 'District of Columbia'
df_predict.index = state_names

# If tweets matched by state: color-code each state via model prediction
from bokeh.palettes import RdBu11 as my_palette

low = 0
high = 1
range_color = np.linspace(low, high, len(my_palette))

p = figure(title="Twitter Results", toolbar_location="left", plot_width=1100, plot_height=700)
for code in states:
    name = states[code]["name"]
    color_ind = np.searchsorted(range_color, df_predict.loc[name].values[0], side='left')
    if color_ind >= len(range_color):
        color_ind -= 1

    # print name, df_predict.loc[name].values[0], color_ind

    p.patch(states[code]["lons"], states[code]["lats"], fill_alpha=0.95, color=my_palette[color_ind],
              line_color="black", line_width=2, line_alpha=0.3)


p.scatter(lon, lat, color='black', alpha=0.6)

legend = generate_colorbar(my_palette, low=low, high=high, plot_width=120, orientation='v')
layout = hplot(p, legend)

output_file("figures/US_map_state.html")

show(layout)


plt.hist(df_predict.values, bins=20)

