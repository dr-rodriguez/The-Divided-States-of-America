# Create a Bokeh map of the tweets

# Load the model previously created
from analysis import Analyzer
from tweetloader import TweetLoader
from bokeh.plotting import figure, show, output_file, vplot, hplot
from bokeh.sampledata.us_states import data as states
from bokeh.models import ColumnDataSource, HoverTool
import reverse_geocoder as rg
import pandas as pd
import numpy as np

# Some global defaults
max_words = 200

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

# Coordinate DataFrame
df_coords = pd.DataFrame(zip(lat, lon), columns=['lat', 'lon'])
df_coords = pd.concat([df_coords, s.tweets['user.screen_name'], s.tweets['text'], s.tweets['state']], axis=1,
                      join_axes=[df_coords.index])

# Group by user
df_user_mode = s.tweets[['user.screen_name', 'predict']].groupby(by='user.screen_name',
                                                           sort=False).agg(lambda x: x.value_counts().index[0])
df_user_count = s.tweets[['user.screen_name', 'predict']].groupby(by='user.screen_name', sort=False).count()
df_state = s.tweets[['user.screen_name', 'state']].groupby(by='user.screen_name',
                                                           sort=False).agg(lambda x: x.value_counts().index[0])
df_user_sum = s.tweets[['user.screen_name', 'predict']].groupby(by='user.screen_name', sort=False).sum()
df_user_mean = s.tweets[['user.screen_name', 'predict']].groupby(by='user.screen_name', sort=False).mean()

# Rename columns and concat
df_user_count.columns = ['count']
df_user_sum.columns = ['sum']
df_user_mean.columns = ['mean']
df_full = pd.concat([df_user_count, df_user_sum, df_user_mean, df_user_mode, df_state], axis=1, join_axes=[df_user_count.index])

# Group by state and get prediction values (mode of user prediction values)
df_state_mean = df_full[['state', 'predict']].groupby(by='state').mean()
df_state_mode = df_full[['state', 'predict']].groupby(by='state', sort=False).agg(lambda x: x.value_counts().index[0])
df_state_count = df_full[['state', 'predict']].groupby(by='state', sort=False).count()
df_state_sum = df_full[['state', 'predict']].groupby(by='state', sort=False).sum()
df_state_mean.columns = ['mean']
df_state_count.columns = ['count']
df_state_sum.columns = ['Trump']
df_predict = pd.concat([df_state_mean, df_state_mode, df_state_sum, df_state_count], axis=1, join_axes=[df_state_mean.index])
df_predict['Clinton'] = df_predict['count'] - df_predict['Trump']
df_predict['poisson'] = np.sqrt(df_predict['count'])

# Rename 'Washington, D.C.' to 'District of Columbia' to match states list
state_names = df_predict.index.tolist()
ind = state_names.index('Washington, D.C.')
state_names[ind] = 'District of Columbia'
df_predict.index = state_names

# Color-coded figure by state and model predictions
p = figure(title="Twitter Results", toolbar_location="left", plot_width=1100, plot_height=700)
for code in states:
    name = states[code]["name"]
    hcount = df_predict.loc[name, 'Clinton']
    tcount = df_predict.loc[name, 'Trump']
    error = df_predict.loc[name, 'poisson']
    if hcount > tcount:
        if abs(hcount - tcount) <= error:
            color = '#92c5de'
        else:
            color = '#0571b0'
    else:
        if abs(hcount - tcount) <= error:
            color = '#f4a582'
        else:
            color = '#ca0020'

    p.patch(states[code]["lons"], states[code]["lats"], fill_alpha=0.8, color=color,
            line_color="black", line_width=2, line_alpha=1)

# Add tweets with on-hover text
data = pd.DataFrame({'lat': lat, 'lon': lon, 'user': s.tweets['user.screen_name'], 'text': s.tweets['text'],
                     'predict': s.tweets['predict'], 'state': s.tweets['state']})
data = data.replace({'predict': {0: 'Hillary', 1: 'Trump'}})
source = ColumnDataSource(data=data)
p.scatter('lon', 'lat', source=source, color='black', alpha=0.8, size=4)
tooltip = {"User": "@user", "Text": "@text", "Prediction": "@predict", 'State': '@state'}
p.add_tools(HoverTool(tooltips=tooltip))

# Axis modifications
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# TODO: Add axes labels

output_file("figures/US_map_state.html")

show(p)

# Compare some tweets
pd.set_option('max_colwidth', 200)
print('Hillary Matches')
print s.tweets[s.tweets['predict'] == 0]['text'].sample(10)  # Hillary
print('Trump Matches')
print s.tweets[s.tweets['predict'] == 1]['text'].sample(10)  # Trump

# Compare with some states as well
print('Trump Matches in New York')
print s.tweets[(s.tweets['predict'] == 1) & (s.tweets['state'] == 'New York')]['text'].sample(10)  # Trump
pd.reset_option('max_colwidth')

# Checking most prolific tweeters
print('Most prolific tweeters:')
df_full[['count', 'mean', 'predict']]\
    .replace({'predict': {0: 'Hillary', 1: 'Trump'}})\
    .sort_values(by='count', ascending=False)\
    .head(15)

# Checking a user's tweets
user_check = 'WashPress'
print("Checking {}'s tweets".format(user_check))
print s.tweets[s.tweets['user.screen_name'] == user_check][['text', 'predict']]\
    .replace({'predict': {0: 'Hillary', 1: 'Trump'}})
