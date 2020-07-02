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

def spacy_binary_text_classifier(input_df, input_label, architecture, epoch_num, batch_size, save_model = 0):
	''' enable binary text classification pipeline using spacy
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: input_label where the labels are in a list of strings .i.e. ['ham','spam']
	input3: spacy architecture to use, options are ensemble, simple_cnn, bow
	input4: epoch_num: number of epochs to run 
	input5: batch_size: size of batch during training
	input6: save the model meta in .py directory, defaults to 0'''


	nlp = spacy.blank("en")
	# Create the TextCategorizer with exclusive classes and "bow" architecture
	textcat = nlp.create_pipe(
	              "textcat",
	              config={
	                "exclusive_classes": True,
	                "architecture": "{}".format(architecture)})

	# Add the TextCategorizer to the empty model
	nlp.add_pipe(textcat)

	# Add labels to text classifier
	textcat.add_label("{}".format(input_label[0]))
	textcat.add_label("{}".format(input_label[1]))


	train_texts = input_df['text'].values
	train_labels = [{'cats': {'{}'.format(input_label[0]): label == '{}'.format(input_label[0]),
	                          '{}'.format(input_label[1]): label == '{}'.format(input_label[1])}} 
	                for label in input_df['label']]


	train_data = list(zip(train_texts, train_labels))

	random.seed(1)
	spacy.util.fix_random_seed(1)
	optimizer = nlp.begin_training()

	losses = {}
	for epoch in range(epoch_num):
	    random.shuffle(train_data)
	    # Create the batch generator with batch size = 8
	    batches = minibatch(train_data, size = batch_size)
	    # Iterate through minibatches
	    for batch in batches:
	        # Each batch is a list of (text, label) but we need to
	        # send separate lists for texts and labels to update().
	        # This is a quick way to split a list of tuples into lists
	        texts, labels = zip(*batch)
	        nlp.update(texts, labels, sgd=optimizer, losses=losses)
	    print(losses)

	if save_model: nlp.to_disk("./")
