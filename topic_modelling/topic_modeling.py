#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code modified by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@


# utility functions for common nlp topic modelling tasks


# def libs
import os
import pickle
import nltk
import re
import string
import spacy
import scipy.sparse
import gensim
import pyLDAvis.gensim
import gensim.corpora as corpora
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from matplotlib import pyplot as plt
from gensim import matutils, models
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from spacy.lang.en import English


# init
spacy.load('en')
parser = English()

    
def tokenize(text):
    '''tokenize the input string'''
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def prepare_text_for_lda(text, removable_words):
    ''' prepare text for LDA analysis'''
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4 if token not in removable_words]
    return tokens

def get_lda_topics(input_str, removable_words, num_of_topics, num_of_passes, save_models = 0, dispaly = 0):
    '''run LDA topic modelling on an input string
    input1 : input string
    input2 : list of removable words 
    input3 : number of topics to explore
    input4 : number of LDA passes
    input5 : to save the model meta or not, deafault to 0
    input6 : to display the topic model using pyLDAvis, throws and error if save_models != dispaly 
    output : dict of topics'''

    if save_models != dispaly:

        raise ValueError('save_models and dispaly parameters should be the same.')
        os.abort()

    #Prepare data for LDA analysis
    data = prepare_text_for_lda(input_str, removable_words)
    text_data = [[data[i]] for i in range(0,len(data))]
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    # LDA Model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_of_topics, id2word = dictionary, passes = num_of_passes)
    topics = ldamodel.print_topics(num_words=4)

    for topic in topics:
        print(topic)

    if save_models:
        pickle.dump(corpus, open('topic_modeling_corpus.pkl', 'wb'))
        dictionary.save('topic_modeling_dictionary.gensim')
        ldamodel.save('topic_modeling_model.gensim')

    if dispaly:
        # load models
        dictionary = gensim.corpora.Dictionary.load('topic_modeling_dictionary.gensim')
        corpus = pickle.load(open('topic_modeling_corpus.pkl', 'rb'))
        lda = gensim.models.ldamodel.LdaModel.load('topic_modeling_model.gensim')
        lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
        pyLDAvis.show(lda_display)

    return topics