# Quick plot on when I gathered the data
from tweetloader import TweetLoader
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import matplotlib.patches as mpatches


def count_and_plot(raw, ax, start='2016-1-1', end='2016-6-24', freq='D', color='blue'):
    df = raw.copy()
    df.index = pd.DatetimeIndex(df['created_at'])
    df = df.sort_index()
    rng = pd.date_range(start, end, freq=freq)
    counts = []
    for i in range(len(rng) - 1):
        num = df['id'][rng[i]:rng[i + 1]].count()
        counts.append(num)
    ax.bar(rng[1:], counts, color=color, lw=0, alpha=0.6)
    return df


h = TweetLoader('HillaryClinton', track_location=False)
h.load()

t = TweetLoader('realDonaldTrump', track_location=False)
t.load()

bs = TweetLoader('BernieSanders', filename='sanders.json', track_location=False)
bs.load()

s = TweetLoader(filename='search.json', track_location=True)
s.load()

# Prepare plot
fig, ax = plt.subplots()

# Chart tweets with time
startdate = '2016-4-30'
enddate = '2016-6-30'
freq = 'D'
count_and_plot(h.tweets, ax, start=startdate, end=enddate, freq=freq, color='blue')
count_and_plot(t.tweets, ax, start=startdate, end=enddate, freq=freq, color='red')
count_and_plot(s.tweets, ax, start=startdate, end=enddate, freq=freq, color='black')

# Plot formatting
ax.set_ylabel('Number of Tweets')
ax.xaxis.set_minor_locator(dates.DayLocator(interval=15))
ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
ax.xaxis.grid(True, which="major")
ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_major_formatter(dates.DateFormatter('\n%b\n%Y'))
# ax.set_yscale("log")
plt.tight_layout()

# Make legend
red = mpatches.Patch(color='red', label='Donald Trump', alpha=0.6)
blue = mpatches.Patch(color='blue', label='Hillary Clinton', alpha=0.6)
black = mpatches.Patch(color='black', label='Search Results', alpha=0.6)
plt.legend(handles=[red, blue, black], loc='best')

plt.savefig('figures/tweets_time.png')
