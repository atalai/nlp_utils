#   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#   **Code altered by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# utility functions for common nlp EDA tasks
# heavily use case dependent so not a lot of room for generic utility functions
# will have to add new approaches/methods if needed


# def libs
import re
import nltk
import spacy
import urllib.request
import string
import pickle
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from spacy.matcher import PhraseMatcher
from spacy.util import minibatch
from spacy.lang.en import English
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS

nlp = spacy.load('en')


def plot_wiki_wordcount(url_link, top = 20):
    '''plot a graph with the most common words in a wiki page
    input1: link to wiki page
    input2: number of displayed words, deafult = 20'''

    response =  urllib.request.urlopen(url_link)
    html = response.read()
    soup = BeautifulSoup(html, "lxml")

    text = soup.get_text(strip = True)
    text = text.lower()
    garbage = ['=','-','.','_','1,\\n', 'would', 'also']

    for i in garbage:

        text = text.replace(i, "")

    text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = WordNetLemmatizer().lemmatize(text)

    tokens = [t for t in text.split()]
    sr = stopwords.words('english')
    clean_tokens = tokens[:]

    for token in tokens:
        if token in stopwords.words('english'):
            
            clean_tokens.remove(token)

    freq = nltk.FreqDist(clean_tokens)
    freq.plot(top, cumulative=False)


def plot_wordcloud(input_str, more_stopwords = ['to', 'from'], title_name = 'N/A'):
    '''plot simple WordCloud
    input1 : input string
    input2: list of stopwords to omit in the cloud
    input3 : title for plot, default = no title'''
    stop_words = more_stopwords + list(STOPWORDS)
    wordcloud = WordCloud(stopwords = stop_words).generate(input_str)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title_name)
    plt.show()









