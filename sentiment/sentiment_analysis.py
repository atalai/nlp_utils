#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code modified by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@


# utility functions for common nlp blind sentiment analysis task


# def libs
import pickle
import nltk
import re
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from matplotlib import pyplot as plt

# def funcs
def raw_sentiment(data_frame, column_name):
	'''add polarity and subjectivity scores in two columns to raw text from a corpus which is in 
	data frame format
	input1: data_frame
	input2: column name where the raw text lives
	output: dataframe with two additional polarity and sensitivity columns'''

	pol = lambda x: TextBlob(x).sentiment.polarity
	sub = lambda x: TextBlob(x).sentiment.subjectivity
	data_frame['polarity'] = data_frame[column_name].apply(pol)
	data_frame['subjectivity'] = data_frame[column_name].apply(sub)

	return data_frame

def plot_sentiment_over_time(input_str, num_of_segments):
	''' plot sentiment over time by dividing input string into num_of_segments text_parts
	and perfroming textblob polarity/sensitivity on each section
	input1 : input string
	input2 : num of parts to analyze
	output : plot'''

	text = input_str
	num_of_parts = num_of_segments
	text_parts = list(map(''.join, zip(*[iter(text)]*int(len(text)/num_of_parts))))

	polarity_scores = [TextBlob(text_parts[i]).sentiment.polarity for i in range(0,num_of_parts)]
	subjectivity_scores = [TextBlob(text_parts[i]).sentiment.subjectivity for i in range(0,num_of_parts)]

	polarity_avg = sum(polarity_scores)/len(polarity_scores)
	subjectivity_avg = sum(subjectivity_scores)/len(subjectivity_scores)


	plt.plot(polarity_scores, color = 'red')
	plt.plot(subjectivity_scores, color = 'blue')
	plt.axhline(y = polarity_avg, color='red', linestyle='--', label = 'polarity_avg')
	plt.axhline(y = subjectivity_avg, color='blue', linestyle='--', label = 'subjectivity_avg')
	plt.title('Polarity/Subjectivity over time')
	plt.xlabel('Segment')
	plt.ylabel('Score')
	plt.legend(loc = 'upper right');
	plt.show()

