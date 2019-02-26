#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyzotero import zotero
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime
from rake_nltk import Rake


# In[2]:


import re
import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix


# In[3]:


from matplotlib import rc

rc(
    "font",
    **{
        "family": "sans-serif",
        "sans-serif": ["Helvetica"],
        "monospace": ["Inconsolata"],
        "serif": ["Baskerville"],
    }
)

rc("text", **{"usetex": True})
rc("text", **{"latex.preamble": "\\usepackage{sfmath}"})
rc("hatch", **{"linewidth": 0.45})


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

# plot constants
dotsize = 2.
linewidth = .5
fontsize = 12


# In[5]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# # See If You Can Load the Dataframe From a Saved Version
# 

# In[7]:


try:
    df = pd.read_pickle("CCA_processed.pickle")
except FileNotFoundError:
    print("Downloading references from Zotero instead")
    zot = zotero.Zotero(436, "user", "qmf7yjuu8uegfsa3p4p7zllk")
    collections = zot.collections()
#     for c in collections:
#     print(c["data"]["name"], c["data"]["key"])
    cca = zot.everything(zot.collection_items_top("96MNUEBN"))
    df = pd.DataFrame(
        {
            "title": [itm["data"]["title"] for itm in cca],
            "date": [itm["meta"]["parsedDate"][0:4] for itm in cca],
            "abstract": [itm["data"]["abstractNote"] for itm in cca],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["abstract_clean"] = df.apply(lambda x: cleanup(x.abstract), axis=1)
df.head()


# # Utility Functions
# 

# In[8]:


def cleanup(txt, stop_words):
    """ Clean up abstract text for statistics """
    text = re.sub("[^a-zA-Z]", " ", txt)
    # Convert to lowercase
    text = text.lower()
    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.split()
    # Stemming
    ps = PorterStemmer()
    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = " ".join(text)
    return text


# Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


# Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2, 2), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


# Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3, 3), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuple of feature, score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def custom_bars(series, cmap):
    data_colour = [x / max(series) for x in series]
    my_cmap = plt.cm.get_cmap(cmap)
    colours = my_cmap(data_colour)
    return colours


# In[9]:


# earliest and latest date criteria
start_date = datetime(2005, 1, 1)
end_date = datetime(2018, 1, 1)
mask = (df["date"] >= start_date) & (df["date"] <= end_date)


# In[92]:


df.to_pickle("CCA_processed.pickle")


# # Graph Research Effort Over Time

# In[10]:


plt.clf()
fig, ax = plt.subplots(1, figsize=(8.0, 6.0), dpi=255)

ax.grid(b=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
ax.set_facecolor("none")
df.loc[mask].date.groupby(df.date.dt.year).agg({"count"}).plot(
    ax=ax,
    color='black',
    alpha=0.85,
#     title="Publications by Year",
    fontsize=fontsize,
)
lines = [2011, 2012, 2016, 2017, 2018]
for line in lines:
    plt.axvline(x=line, color="black", lw=0.5, linestyle="--")
plt.axvspan(2005, 2011, facecolor='grey', alpha=0.3)
plt.axvspan(2012, 2016, facecolor='grey', alpha=0.3)
plt.axvspan(2017, 2018, facecolor='grey', alpha=0.3)
leg = ax.legend(["Count of Publications"], fontsize=9)
ax.set_ylabel("Publications", fontsize=fontsize)
plt.title("Research Effort, 1992 - 2018")
ax.set_xlabel("Year", fontsize=fontsize)
plt.tight_layout()
plt.savefig("time.png", dpi=300)
plt.show()


# # Preliminary Text Analysis

# In[81]:


r = Rake()


# In[137]:


key_phrases = []
for idx, row in df.iterrows():
    r.extract_keywords_from_text(row.abstract)
    # extract top 5 phrases
    key_phrases.append(r.get_ranked_phrases_with_scores()[0:9])


# In[150]:


key_phrases[10]


# In[184]:


df["word_count"] = df["abstract"].apply(lambda x: len(str(x).split(" ")))
df[["abstract", "word_count"]].head()


# In[185]:


# descriptive statistics
df.word_count.describe()


# In[187]:


# 20 most frequently-occurring words in the abstracts
freq = pd.Series(" ".join(df["abstract"]).split()).value_counts()[:20]
freq


# In[188]:


# 20 least-common words
frequ = pd.Series(" ".join(df["abstract"]).split()).value_counts()[-20:]
frequ


# # Text Pre-Processing
# 

# In[11]:


# Create a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
# Create a list of custom stopwords
new_words = [
    "using",
    "show",
    "result",
    "large",
    "also",
    "iv",
    "one",
    "two",
    "new",
    "previously",
    "shown",
]
# these are the search keywords
new_words.extend(["climate", "change", "adaptation", "engagement", "public", "citizen"])
# these are metadata words, we definitely don't want these
new_words.extend(
    [
        "paper",
        "ltd",
        "limited",
        "right",
        "reserved",
        "elsevier",
        "john",
        "wiley",
        "sons",
        "article",
        "please",
        "visit",
        "case",
        "study",
    ]
)
stop_words = stop_words.union(new_words)

df["abstract_clean"] = df.apply(lambda x: cleanup(x.abstract, stop_words), axis=1)


# In[12]:


# build a word cloud
plt.clf()
fig, ax = plt.subplots(1, figsize=(8.0, 6.0), dpi=100)
wordcloud = WordCloud(
    background_color="white",
    width=1800,
    height=1200,
    stopwords=stop_words,
    max_words=100,
    max_font_size=75,
    random_state=42,
    colormap="BuPu",
).generate(str(df.abstract_clean))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloud.png", dpi=300)
plt.show()


# In[13]:


cv = CountVectorizer(
    max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 3)
)
X = cv.fit_transform(df.abstract_clean)


# In[14]:


list(cv.vocabulary_.keys())[:10]


# In[15]:


# Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(df.abstract_clean, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns = ["Word", "Freq"]


# # Graph Most Frequent Words

# In[16]:


plt.clf()
fig, ax = plt.subplots(1, figsize=(6.0, 4.0), dpi=255)

c = custom_bars(top_df.Freq, "Greys")

top_df.plot.bar(
    ax=ax, x="Word", y="Freq", fontsize=6, color=c, edgecolor="#282828", lw=0.2
)
leg = ax.legend(["Word Frequency"], fontsize=6)
plt.title("20 Most Frequent Words")
ax.grid(b=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor("none")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("words_single.png", dpi=300)
plt.show()


# In[17]:


top2_words = get_top_n2_words(df.abstract_clean, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns = ["Bigram", "Freq"]


# In[18]:


plt.clf()
fig, ax = plt.subplots(1, figsize=(6.0, 4.0), dpi=255)

c = custom_bars(top2_df.Freq, "Greys")

top2_df.plot.bar(
    ax=ax, x="Bigram", y="Freq", fontsize=5, color=c, edgecolor="#282828", lw=0.2
)
plt.title("20 Most Frequent Bigrams")
leg = ax.legend(["Bigram Frequency"], fontsize=6)
ax.grid(b=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor("none")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bigrams.png", dpi=300)
plt.show()


# In[19]:


top3_words = get_top_n3_words(df.abstract_clean, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns = ["Trigram", "Freq"]


# In[20]:


plt.clf()
fig, ax = plt.subplots(1, figsize=(6.0, 4.0), dpi=255)

c = custom_bars(top3_df.Freq, "Greys")

top3_df.plot.bar(
    ax=ax, x="Trigram", y="Freq", fontsize=5, color=c, edgecolor="#282828", lw=0.2
)
plt.title("20 Most Frequent Trigrams")
leg = ax.legend(["Trigram Frequency"], fontsize=9)
ax.grid(b=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor("none")
plt.xticks(rotation=80)
plt.tight_layout()
plt.savefig("trigrams.png", dpi=300)
plt.show()


# In[21]:


tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)
# get feature names
feature_names = cv.get_feature_names()

# fetch document for which keywords needs to be extracted
doc = df.abstract_clean.iloc[50]

tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
# sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())
# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 5)


# In[22]:


print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k, keywords[k])


# In[23]:


df[df.abstract_clean.str.contains("rise")].date.groupby(df.date.dt.year).agg({"count"})


# In[24]:


df[df.abstract_clean.str.contains("rise")].date.groupby(df.date.dt.year).agg(
    {"count"}
).sum()


# In[25]:


df.groupby(df.date.dt.year).agg({"count"})


# # Geographic Classification

# In[26]:


df_countries = pd.read_excel("/Users/sth/OneDrive - TCDUD.onmicrosoft.com/WIREs 2018/summaries.xlsx")
df_countries.head()


# In[27]:


df_countries["Year"] = pd.to_datetime(df_countries["Year"], format="%Y")
df_countries.dropna(inplace=True)
uniq = set(list(df_countries.Country))


# In[31]:


# map countries to areas
mappings = {
    "AUS": "AUS / NZ",
    "NZ": "AUS / NZ",
    "BEL": "Europe",
    "BGDSH": "Rest of World",
    "BRA": "Rest of World",
    "CAN": "North America",
    "CH": "Europe",
    "CHIL": "Rest of World",
    "DE": "Europe",
    "DK": "Europe",
    "FI": "Europe",
    "FR": "Europe",
    "HK": "Rest of World",
    "IE": "Europe",
    "IND": "Rest of World",
    "INDNO": "Rest of World",
    "IRN": "Rest of World",
    "IT": "Europe",
    "JAP": "Rest of World",
    "KEN": "Rest of World",
    "MAC": "Rest of World",
    "MAL": "Europe",
    "NIG": "Rest of World",
    "NL": "Europe",
    "NO": "Europe",
    "POR": "Europe",
    "PRC": "Rest of World",
    "ROC": "Rest of World",
    "SA": "Rest of World",
    "SING": "Rest of World",
    "SWE": "Europe",
    "TWN": "Rest of World",
    "UGND": "Rest of World",
    "UK": "UK",
    "URG": "Rest of World",
    "US": "North America",
    "VNM": "Rest of World"
}


# In[32]:


# assign area by looking up countries
df_countries.Country = df_countries['Country'].astype('category')
df_countries['Area'] = df_countries['Country'].apply(lambda x: mappings.get(x))
df_countries.Area = df_countries['Area'].astype('category')
df_countries.head()


# In[49]:


plt.clf()
fig, ax = plt.subplots(1, figsize=(6.0, 4.0), dpi=255)

df_countries.set_index('Year').groupby(['Year', 'Area']).count()['Country'].unstack().plot(
    kind='bar',
    stacked=True,
    fontsize=5,
    color='#ffffff',
    edgecolor="#282828",
    lw=0.2,
    ax=ax,
)
plt.title("Research Effort Grouped By Area")
ax.grid(b=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor("none")
# manually create the tick labels, by pulling in the correct number of years
# and the lowest year value, formatted as a string
periods = len(df_countries.groupby('Year'))
ticklabels = pd.Series(
    pd.date_range(df_countries.Year.min().strftime("%Y"), freq='Y', periods=periods)
).dt.strftime('%Y')
ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
fig.autofmt_xdate()

bars = ax.patches
# the choice of hatches could be automatically derived from the number of Areas
# doing it manually gives better control over adjacent patterns (so "/" isn't adjacent to "x")
ha = [h * periods for h in '-/O.x']
hatches = ''.join(ha)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch * 6)

ax.set_ylabel("Publications", fontsize=fontsize)
ax.legend(loc='upper left', ncol=1, fontsize=5)

plt.tight_layout()
plt.savefig("areas.png", dpi=300)
plt.show()


# # Messing with Fonts

# In[13]:


matplotlib.font_manager.FontProperties(fname='/Users/sth/Library/Fonts/Inconsolata-Regular.ttf').get_name()


# In[5]:


matplotlib.get_cachedir()


# In[6]:


matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')


# In[ ]:




