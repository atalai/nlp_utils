#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@


# utility functions for common nlp text classification tasks 


# def libs
import sys
import spacy
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from spacy.util import minibatch
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

# def funcs
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

def linearsvc_binary_text_classifier(input_df, spacy_model, test_split, iterations):
	''' enable binary text classification pipeline using linear svc and spacy english models
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: spacy model, has two options 'basic' and 'advanced'
	input3: test_split size ranges from (0,1)
	input4: number of iteration fro the SVC model
	output: model object'''

	if spacy_model == 'basic': 	nlp = spacy.load('en')
	if spacy_model == 'advanced': nlp = spacy.load('en_core_web_lg')

	# prepare dataset
	doc_vectors = np.array([nlp(text).vector for text in input_df.text])

	# Linear Support Vector Classification
	X_train, X_test, y_train, y_test = train_test_split(doc_vectors, input_df.label, test_size = test_split, random_state=1993)    
	svc = LinearSVC(random_state=1993, dual = False, max_iter = iterations)
	svc.fit(X_train, y_train)
	print (svc.score(X_test, y_test))

	return svc

def svc_binary_text_classifier(input_df, spacy_model, test_split, kernel_type, probability_flag = False):
	''' enable binary text classification pipeline using non_linear svc and spacy english models
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: spacy model, has two options 'basic' and 'advanced'
	input3: test_split size ranges from (0,1)
	input4: kernel type for SVC model : {‘poly’, ‘rbf’, ‘sigmoid’}
	input5: produce probability on .predict, defaults to False
	output: model object'''

	if spacy_model == 'basic': 	nlp = spacy.load('en')
	if spacy_model == 'advanced': nlp = spacy.load('en_core_web_lg')

	# prepare dataset
	doc_vectors = np.array([nlp(text).vector for text in input_df.text])

	# no-Linear Support Vector Classification
	X_train, X_test, y_train, y_test = train_test_split(doc_vectors, input_df.label, test_size = test_split, random_state = 1993) 
	svc = SVC(kernel = kernel_type, probability = probability_flag)
	svc.fit(X_train, y_train)
	print (svc.score(X_test, y_test))

	return svc


#linearsvc_binary_text_classifier(train_df, 'basic', 0.1, 50)
#svc_binary_text_classifier(train_df, 'advanced', 0.1, 'rbf', True)

def rf_binary_text_classifier(input_df, spacy_model, test_split, tree_count):
	''' enable binary text classification pipeline using random forest and spacy english models
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: spacy model, has two options 'basic' and 'advanced'
	input3: test_split size ranges from (0,1)
	input4: kernel type for SVC model : {‘poly’, ‘rbf’, ‘sigmoid’}
	input5: produce probability on .predict, defaults to False
	output: model object'''

	if spacy_model == 'basic': 	nlp = spacy.load('en')
	if spacy_model == 'advanced': nlp = spacy.load('en_core_web_lg')

	# prepare dataset
	doc_vectors = np.array([nlp(text).vector for text in input_df.text])

	# random forest Classification
	X_train, X_test, y_train, y_test = train_test_split(doc_vectors, input_df.label, test_size = test_split, random_state = 1993) 
	rfc = RandomForestClassifier(n_estimators = tree_count, random_state = 1993)
	rfc.fit(X_train, y_train)
	print (rfc.score(X_test, y_test))

	return rfc


def gb_binary_text_classifier(input_df, spacy_model, test_split, boosting_num = 100, lr = 0.05):
	''' enable binary text classification pipeline using gradient boosting and spacy english models
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: spacy model, has two options 'basic' and 'advanced'
	input3: test_split size ranges from (0,1)
	input4: number of boosting stages to use
	input5: learning rate
	output: model object'''

	if spacy_model == 'basic': 	nlp = spacy.load('en')
	if spacy_model == 'advanced': nlp = spacy.load('en_core_web_lg')

	# prepare dataset
	doc_vectors = np.array([nlp(text).vector for text in input_df.text])

	# gradient boosting Classification
	X_train, X_test, y_train, y_test = train_test_split(doc_vectors, input_df.label, test_size = test_split, random_state = 1993) 
	GBoost = GradientBoostingClassifier(n_estimators = boosting_num, learning_rate = lr,
                                   max_depth = 4, max_features='sqrt',
                                   min_samples_leaf = 15, min_samples_split=10, 
                                   random_state = 1993)
	GBoost.fit(X_train, y_train)
	print (GBoost.score(X_test, y_test))

	return GBoost


def lgb_binary_text_classifier(input_df, spacy_model, test_split, leaves_num = 100, lr = 0.05):
	''' enable binary text classification pipeline using light gradient boosting and spacy english models
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: spacy model, has two options 'basic' and 'advanced'
	input3: test_split size ranges from (0,1)
	input4: number of boosting stages to use
	input5: learning rate
	output: model object'''

	if spacy_model == 'basic': 	nlp = spacy.load('en')
	if spacy_model == 'advanced': nlp = spacy.load('en_core_web_lg')

	# prepare dataset
	doc_vectors = np.array([nlp(text).vector for text in input_df.text])

	# light gradient boosting Classification
	X_train, X_test, y_train, y_test = train_test_split(doc_vectors, input_df.label, test_size = test_split, random_state = 1993) 
	model_lgb = lgb.LGBMClassifier(
	                              objective = 'binary',num_leaves = leaves_num,
	                              learning_rate = lr, n_estimators = 4420,
	                              max_bin = 55, bagging_fraction = 0.8,
	                              bagging_freq = 5, feature_fraction = 0.2319,
	                              feature_fraction_seed=9, bagging_seed = 9,
	                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 1)
	model_lgb.fit(X_train, y_train)
	print (model_lgb.score(X_test, y_test))

	return model_lgb


def ensemble_binary_text_classifier(input_df, spacy_model, test_split):
	''' enable ensemble binary text classification pipeline using major classifiers and spacy english models
	should only be used to get a feel for the data
	input1: input dataframe where the format is data_label, content and the column names are label and text
	input2: spacy model, has two options 'basic' and 'advanced'
	input3: test_split size ranges from (0,1)
	output: accuracy of classifications as a dictionary'''

	if spacy_model == 'basic': 	nlp = spacy.load('en')
	if spacy_model == 'advanced': nlp = spacy.load('en_core_web_lg')

	# prepare dataset
	doc_vectors = np.array([nlp(text).vector for text in input_df.text])
	# prepare trai and test datasets 
	X_train, X_test, y_train, y_test = train_test_split(doc_vectors, input_df.label, test_size = test_split, random_state = 1993)

	model_lgb = lgb.LGBMClassifier(
	                              objective = 'binary',num_leaves = 100,
	                              learning_rate = 0.05, n_estimators = 4420,
	                              max_bin = 55, bagging_fraction = 0.8,
	                              bagging_freq = 5, feature_fraction = 0.2319,
	                              feature_fraction_seed=9, bagging_seed = 9,
	                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 1)

	model_lgb.fit(X_train, y_train)

	GBoost = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.05,
                                   max_depth = 4, max_features = 'sqrt',
                                   min_samples_leaf = 15, min_samples_split = 10, 
                                   random_state = 1993)
	GBoost.fit(X_train, y_train)

	rfc = RandomForestClassifier(n_estimators = 100, random_state = 1993)
	rfc.fit(X_train, y_train)

	svc = SVC(kernel = 'rbf', probability = False)
	svc.fit(X_train, y_train)

	linear_svc = LinearSVC(random_state=1993, dual = False, max_iter = 100)
	linear_svc.fit(X_train, y_train)


	model_lgb_score = model_lgb.score(X_test, y_test)
	gboost_lgb_score = GBoost.score(X_test, y_test)
	rf_score = rfc.score(X_test, y_test)
	svc_score = svc.score(X_test, y_test)
	linear_svc_score = linear_svc.score(X_test, y_test)

	results = {'linear_svc': linear_svc_score, 'svc': svc_score,'random_forest': rf_score, 
				'gradient_boosting': gboost_lgb_score, 'light_gradient_boosting': model_lgb_score}

	print (results)
	return results
