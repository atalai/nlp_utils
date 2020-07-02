#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@


# utility functions for common nlp topic modelling tasks


# def libs
import os
import sys
import pandas as pd
import numpy as np
import spacy
import random
from spacy.util import minibatch
sys.path.insert(1, '/Users/sahandtalai/Desktop/nlp_utils/pre_process')
import clean_data
from clean_data import lower_case, remove_brackets, lemmatize_str, remove_letters, remove_nline, remove_numbers, remove_prefix_b
from clean_data import remove_punct, remove_quotation, remove_single_char_start, remove_special_char, remove_numbers
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# def funcs
def pre_process_text(text):
	'''make a custom pre-processing function based on the clean_data master function'''
	txt = lower_case(text)
	txt = remove_brackets(txt)
	txt = remove_nline(txt)
	txt = remove_punct(txt)
	txt = remove_quotation(txt)
	txt = remove_special_char(txt)
	txt = remove_letters(txt)
	txt = remove_numbers(txt)
	txt = lemmatize_str(txt)
	return txt

def make_balanced_dataset(input_df, class_size):
	''' make a balanced dataset of binary labels given class_size and generate independent train and validation datasets
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: class size is the number of training cases for each category
	output: shuffeled pandas dataframe for train and validation'''

	# get list of unique labels in dataframe
	label_count = []
	labels = input_df.label.unique()
	for label in labels:
		input_df_temp = input_df[input_df['label'] == label]
		label_count.append([label, len(input_df_temp)])

	# make a balanced training data set 
	minority_label = sorted(label_count, key = lambda x:x[1])[0]
	majority_label = sorted(label_count, key = lambda x:x[1])[1]

	if class_size > int(minority_label[1]):
		print ('Class size should be lower than minority class size!')
		raise ValueError 
		os.abort()

	temp_data = input_df[input_df['label'] == str(minority_label[0])]
	minority_df = temp_data.sample(frac=1)
	minority_df_train = minority_df.head(class_size)
	minority_df_val = minority_df.tail(len(minority_df) - class_size)

	temp_data = input_df[input_df['label'] == str(majority_label[0])]
	majority_df = temp_data.sample(frac=1)
	majority_df_train = majority_df.head(class_size)
	majority_df_val = majority_df.tail(len(majority_df) - class_size)

	frames = [minority_df_train, majority_df_train]
	merged_df = pd.concat(frames)
	merged_df = merged_df.sample(frac=1)
	merged_train = merged_df.reset_index()

	frames = [minority_df_val, majority_df_val]
	merged_df = pd.concat(frames)
	merged_df = merged_df.sample(frac=1)
	merged_val = merged_df.reset_index()

	return merged_train, merged_val


def spacy_binary_text_classifier(input_df, input_label, architecture, epoch_num, batch_size, save_model = 0):
	''' enable binary text classification pipeline using spacy
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: input_label where the labels are in a list of strings .i.e. ['ham','spam']
	input3: spacy architecture to use, options are ensemble, simple_cnn, bow
	input4: epoch_num: number of epochs to run 
	input5: batch_size: size of batch during training
	input6: save the model meta in .py directory, defaults to 0
	output: return the model that can be used inside the script'''

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

	return nlp



# read data
spam = pd.read_csv("spam.csv", encoding = "latin-1")
spam = spam[['v1', 'v2']]
data = spam.rename(columns = {'v1': 'label', 'v2': 'text'})


train_df, val_df = make_balanced_dataset(data, 700)

# apply preprocessing function on data
func = lambda x: pre_process_text(x)
train_df['text'] = train_df['text'].apply(func)
val_df['text'] = val_df['text'].apply(func)





## spacy binary classification
# nlp = spacy_binary_text_classifier(train_df, ['ham', 'spam'], 'simple_cnn', 5, 64, 0)


# Linear Support Vector Classification
nlp = spacy.load('en_core_web_lg')
with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in train_df.text])


X_train, X_test, y_train, y_test = train_test_split(doc_vectors, train_df.label, test_size=0.1, random_state=1)    
svc = LinearSVC(random_state=1, dual=False, max_iter=100)
svc.fit(X_train, y_train)
print (svc.score(X_test, y_test))




# validation = val_df['text'].tolist()

# # apply the same preprocessing steps as the training stage
# func = lambda x: pre_process_text(x)
# texts = list(map(func, validation))


# docs = [nlp.tokenizer(text) for text in texts]    
# # Use textcat to get the scores for each doc
# textcat = nlp.get_pipe('textcat')
# scores, _ = textcat.predict(docs)
# predicted_labels = scores.argmax(axis=1)

# document = 0
# for label in predicted_labels:

# 	print (docs[document],'--->' ,textcat.labels[label])
# 	document = document + 1
# 	print ('------------------')