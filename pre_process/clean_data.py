#   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#   **Code modified/enriched by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# utility functions for common nlp pre_processing task


# def libs
import pickle
import nltk
import re
import string
import spacy
import pprint 
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from spacy.matcher import PhraseMatcher
from spacy.util import minibatch
from spacy.lang.en import English
from nltk import word_tokenize, pos_tag
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim import matutils, models


nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en')

# def funcs 
def lower_case(input_str):
    ''' make string all lowrcase'''
    return input_str.lower()

def remove_brackets(input_str):
    '''remove text inside brackets'''
    return re.sub('\[.*?\]', '', input_str)

def remove_punct(input_str):
    '''remove punctuation'''
    return re.sub('[%s]' % re.escape(string.punctuation), '', input_str)

def remove_numbers(input_str):
    '''remove words with numbers in them. removes seven7 and single digit with spaces'''
    return re.sub('\w*\d\w*', '', input_str)

def remove_nline(input_str):
    '''remove \n from strings'''
    return re.sub('\n', '', input_str)

def remove_special_char(input_str):
    '''remove special characters'''
    return re.sub(r'\W', ' ', input_str)

def remove_quotation(input_str):
    '''remove any quotation marks'''
    return re.sub('[‘’“”…]', '', input_str)

def remove_letters(input_str):
    '''remove signle letter, often used after previous cleaning methods have been applied'''
    return re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', input_str)

def remove_single_char_start(input_str):
    '''remove single characters from the start'''
    return re.sub(r'\^[a-zA-Z]\s+', ' ', input_str) 

def remove_prefix_b(input_str):
    ''' Removing prefixed 'b'''
    return re.sub(r'^b\s+', '', input_str)

def remove_accented(input_str):
    """remove accented characters from text, e.g. café, seems to wreck spacy conda installation"""
    return unidecode.unidecode(input_str)

def lemmatize_str(input_str):
    '''lemmatize words and phrases in a string'''
    return WordNetLemmatizer().lemmatize(input_str) 

def get_nouns(input_str):
    '''extract only the nouns from a string and return the updated string'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(input_str)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)

def get_nouns_adj(input_str):
    '''extract only the nouns and adjectives from a string and return the updated string'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(input_str)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)

def to_dtm(input_df, text , remove_stop_words = 0):
    ''' create a document term matrix as a dataframe from an initial dataframe where the format is text, lable to label vs tokenized words
    input1 : data frame
    input2 : column name with the raw text (usually preprocessed)
    input3 : remove english stop words in the process, deafult to 0'''
    if remove_stop_words :  cv = CountVectorizer(stop_words='english')
    cv = CountVectorizer()
    data_cv = cv.fit_transform(data_clean[text])
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data_clean.index

    return data_dtm

def remove_stop_punct (input_str, output_mode = 0):
    ''' remove stop words and punctuation in a string
    input1: input string
    input 2: output_mode -- 0:list, 1--> string
    output: list of filtered text items'''
    doc = nlp(input_str)
    text = [token for token in doc if not token.is_stop and not token.is_punct]
    text = [token.lemma_ for token in text]

    if output_mode: text = ' '.join(text)
    return text

def find_match_location(input_str, pattern_list):
    ''' find matching patterns in an input string
    input 1: input string
    input 2: list of patterns to be found 
    output : tuple with the following format : (id, match_start, match_end)''' 
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp(text) for text in pattern_list]
    matcher.add("TerminologyList", None, *patterns)
    text_doc = nlp(input_str) 
    matches = matcher(text_doc)
    return matches


def get_tfidf(input_str):
    '''calculate Term Frequency-Inverse Document Frequency matrix on input_str, returns
    sorted TF_IDF document'''
    result = []
    doc_list = [input_str]
    doc_tokenized = [simple_preprocess(doc) for doc in doc_list]
    dictionary = corpora.Dictionary()
    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
    tfidf = models.TfidfModel(BoW_corpus, smartirs='ntc')

    for doc in tfidf[BoW_corpus]:
       result.append([[dictionary[id], np.around(freq,decimals=2)] for id, freq in doc])

    result = result[0]
    sorted_result = sorted(result, key = lambda x: (x[1]) , reverse = True)

    return sorted_result


def cosine_similarity(a, b):
    '''run cosine text similarity test'''
    return a.dot(b)/np.sqrt(a.dot(a) * b.dot(b))

def text_similarity(str1, str2, enable_large_model = 0):
    '''spacy text similarity metrics'''
    nlp = spacy.load('en')
    if enable_large_model: nlp = spacy.load('en_core_web_lg')
    str1 = nlp(str1).vector
    str2 = nlp(str2).vector

    return cosine_similarity(str1, str2)

def clean_text(input_str):
    ''' pipe various cleaning methods based on use case'''
    text = lower_case(input_str)
    text = remove_brackets(text)
    text = remove_punct(text)
    text = remove_nline(text)
    text = remove_quotation(text)
    text = remove_numbers(text)
    text = lemmatize_str(text)
    cleaned_text = remove_letters(text)

    return cleaned_text
