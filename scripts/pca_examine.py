# Examine PCA in a grid plot
from tweetloader import TweetLoader
from analysis import Analyzer
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Load the tweets
# h = TweetLoader('HillaryClinton')
# t = TweetLoader('realDonaldTrump')
h = TweetLoader('', path='data/backup/', filename='hillary_2016-07-13.json')
t = TweetLoader('', path='data/backup/', filename='trump_2016-07-13.json')
h.load()
t.load()

# Join them together
full_tweets = pd.concat([h.tweets, t.tweets])

# Assign label (second array) for Hillary(0)/Trump(1) tweets
label_array = np.array([0]*len(h.tweets) + [1]*len(t.tweets))

# Run through part of the model to get the PCA results and loading factors
# This is not the full model, just a part of it for illustration purposes
max_words = 50
mod = Analyzer(full_tweets['text'], labels=label_array, max_words=max_words, load_pca=False)

# mod.load_words()
mod.get_words()
mod.create_dtm()
mod.run_pca()

loadings = mod.loadings
loadings.index = ['PC'+str(j+1) for j in range(len(loadings))]

# loadings = loadings.iloc[0:30, :]  # Use only a subset of the data
loadings = loadings.transpose()  # Use rotation

words = loadings.columns.tolist()
pc_names = loadings.index.tolist()

xname = []
yname = []
color = []
alpha = []
for i, pc in enumerate(pc_names):
    for j, word in enumerate(words):
        xname.append(word)
        yname.append(pc)

        alpha.append(min(loadings.iloc[i, j]**2, 0.9)+0.1)  # Transparency is square of loading factor

        # Color denotes sign of loading factor
        if loadings.iloc[i, j] > 0:
            color_to_use = '#5ab4ac'
        else:
            color_to_use = '#d8b365'

        if abs(loadings.iloc[i, j]) < 0.1:
            color_to_use = '#f5f5f5'

        color.append(color_to_use)

source = ColumnDataSource(data=dict(
    xname=xname,
    yname=yname,
    colors=color,
    alphas=alpha,
    count=loadings.values.flatten(),
))

p = figure(title="PCA Loading Factors",
           x_axis_location="above", tools="pan,wheel_zoom,box_zoom,reset,hover,save",
           x_range=words, y_range=list(reversed(pc_names)))

p.plot_width = 1000
p.plot_height = 1000
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "8pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi / 3

p.rect('xname', 'yname', 0.9, 0.9, source=source,
       color='colors', alpha='alphas', line_color=None)

p.select_one(HoverTool).tooltips = [
    ('pc/word', '@yname, @xname'),
    ('factor', '@count'),
]

output_file("figures/pca_factors.html", title="PCA Loading Factors")

show(p)

# Examine via biplot
mod.make_biplot(0, 3, max_arrow=0.2, save='figures/biplot_0_3.png', alpha=0.3, use_sns=True)
mod.make_biplot(4, 15, max_arrow=0.3, save='figures/biplot_4_15.png', alpha=0.3, use_sns=True)
mod.make_biplot(3-1, 4-1, 0.3, alpha=0.2, use_sns=True, save='figures/biplot_3_4.png')
mod.make_biplot(5-1, 13-1, 0.3, alpha=0.3, use_sns=True)

# Grid PCA plot with Seaborn
data = mod.pcscores.iloc[:, 0:4]
data = pd.concat([data, pd.Series(label_array, name='label')], axis=1)
data = data.replace({'label': {0: 'Hillary Clinton', 1: 'Donald Trump'}})
# Normalize range to -1 to 1
for col in data.columns:
    if col == 'label': continue
    data[col] = data[col].apply(lambda x: x*1.0 / (data[col].max() - data[col].min()))
# sns.set_style('whitegrid')
g = sns.pairplot(data=data, hue='label', palette={'Hillary Clinton': 'blue', 'Donald Trump': 'red'},
                 diag_kind='kde', plot_kws=dict(alpha=0.5))
# sns.reset_orig()
