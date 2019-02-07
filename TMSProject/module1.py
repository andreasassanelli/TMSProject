# Text Mining And Search Final Assesment
#
# Nicoli, Paolo         833311
# Sassanelli, Andrea    000000
#
# Submission date 2019-02-08
#
# Excercise dataset: http://qwone.com/~jason/20Newsgroups/

#Load libraries
import os
import nltk
import re
from time import time
import random
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier

#import components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from matplotlib import pyplot as plt

# Import sklearn components and Classifiers
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# KERAS imports
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
import keras_metrics

# Download/Refresh and load NLTK components (Stemmer, English stop-words)
nltk.download("punkt")
pstemmer = PorterStemmer()

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

# Global constants (dataset file locations, etc)
DEBUG = False           #Toggle debug mode
random.seed(1)          #Set to 1 for experimental consistency
# Dataset on FS (source: http://)
datasetRootDir = "c:/datasets/20news-bydate/"
testSetRootDir = datasetRootDir + "20news-bydate-test/"
trainSetRootDir = datasetRootDir + "20news-bydate-train/"

# 20newsgroup dataset-specific preprocessing functions
# source: scikit-learn/sklearn/datasets/twenty_newsgroups.py
#
# Removes headers, citation words and citation marks.
_HEADER_RE = re.compile(r'^[A-Za-z0-9-]*:')
_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')

# Given text in "news" format, strip the headers, by removing everything
# before the first blank line.
# Additionally, drop lines leading with a header-like pattern, ie "Key-Text:"
#
# input: string representation of a whole raw usenet message
# output: string of raw usenet message trimmed of all header lines
def strip_newsgroup_header(text):
    _before, _blankline, after = text.partition('\n\n')
    after_clean = [line for line in after.split('\n')
                  if not _HEADER_RE.search(line)]
    return '\n'.join(after_clean)

# Given text in "news" format, strip lines beginning with the quote
# characters > or |, plus lines that often introduce a quoted section
# (for example, because they contain the string 'writes:'.)
#
# input: string representation of a whole raw usenet message
# output: string of raw usenet message trimmed of all characters preceding quotes
def strip_newsgroup_quoting(text):
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)

# Given text in "news" format, attempt to remove a signature block.
# As a rough heuristic, we assume that signatures are set apart by either
# a blank line or a line made of hyphens, and that it is the last such line
# in the file (disregarding blank lines at the end).
#
# input: string representation of a whole raw usenet message
# output: string of raw usenet message trimmed of some footer lines / signatures
def strip_newsgroup_footer(text):

    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

# --- DATASET LOADING AND PREPROCESSING ROUTINES --- #
# Pre-process raw message by applying tokenization, stopwords removal and stemming
def preprocDocument(document):

    # tokenize
    termlist = word_tokenize(document)

    # drop non alphabetic elements and stopwords, convert to lowercase
    termlist = [elem.lower() for elem in termlist if elem.isalpha() and elem not in stop_words]

    # stemming
    termlist = [pstemmer.stem(w) for w in termlist]

    return termlist

# Load dataset from specified location assuming each class has its own subfolder containing document files.
# Removes header/quote/footer from each document according to boolean flags in <strip_flags> tuple.
# Applies optional pre-processing function <func> to each document individually.
# Returns list of [docID, class, data], one for each document in dataset
def importDataSet(datasetLocation, strip_flags = (True, True, True), func = lambda x: x , verbose = False):

    if verbose: print(datasetLocation)

    # prepare empty list to return
    dataset = list()

    # for each class in sample
    for groupName in os.listdir(datasetLocation):

        # Append the group name to the directory path
        corpusDir = datasetLocation + groupName

        # Get list of documents files
        fileList = os.listdir(corpusDir)

        if verbose: print("%s: %s" % (groupName,len(fileList)))

        # for each fileID in folder
        for docID in fileList:

            # load document as string
            with open( corpusDir + '/' + docID) as fin:
                data = fin.read()

            if DEBUG:
                print("vanilla")
                print(data)
                print("-----------------------------------------------------------------------------------")

            # apply pre-processing functions according to "strips" flags
            if strip_flags[0]: data = strip_newsgroup_header(data)
            if strip_flags[1]: data = strip_newsgroup_quoting(data)
            if strip_flags[2]: data = strip_newsgroup_footer(data)

            if DEBUG:
                print("strip")
                print(data)
                print("---------------------------------------------------------------------------------")

            # apply additional processing if needed
            data = func(data)

            if DEBUG:
                print("end")
                print(data)
                print("---------------------------------------------------------------------------------")

            # append resulting tuple to output dataset
            dataset.append([docID, groupName, data])

            if DEBUG: break
        if DEBUG: break

    return dataset

# --- ENTRY POINT --- #
def main():
    print("Entry Point")

    # Load 20newsgroups-bydate dataset
    # The dataset is split at source in a training and a test subsets
    #
    # Load and preprocess training subset data
    t0 = time()
    raw_train = importDataSet(trainSetRootDir, strip_flags=stripflg, func=preprocDocument, verbose=verbose)
    print("training dataset loaded in %d seconds" % (time() -t0) )
    random.shuffle(raw_train)
    # Load and preprocess testing subset data
    t1 = time()
    raw_test = importDataSet(testSetRootDir, strip_flags=stripflg, func=preprocDocument, verbose=verbose)
    print("test dataset loaded in %d seconds" % (time() -t1) )
    random.shuffle(raw_test)

    # collate corpus
    train_corpus = [ ' '.join(x[2]) for x in raw_train]
    test_corpus = [ ' '.join(x[2]) for x in raw_test]

    train_labels = [x[1] for x in raw_train]
    test_labels = [x[1] for x in raw_test]

    label_names = list(set(test_labels))

    count_vect = CountVectorizer(min_df=min_freq, max_df=max_freq)
    tfidf_trans = TfidfTransformer()
    
    # Our split 
    train_posts = train_corpus
    test_posts = test_corpus
    train_tags = train_labels
    test_tags = test_labels

    # build Document-Term matrix with TF-IDF weights
    X_train = count_vect.fit_transform(train_corpus)
    X_train_tfidf = tfidf_trans.fit_transform(X_train)

    # Classification MultinomialNB
    print("Classification using MultinomialNB")
    clf = MultinomialNB().fit(X_train_tfidf, train_labels)

    X_test = count_vect.transform(test_corpus)
    X_test_tfidf = tfidf_trans.transform(X_test)

    y_predict = clf.predict(X_test_tfidf)

    # Evaluation
    print(metrics.classification_report(test_labels, y_predict, target_names = label_names))
    cmat = metrics.confusion_matrix(test_labels, y_predict)

    print(cmat)
    # KERAS
    

    max_words = 1000
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(train_posts) # only fit on train

    x_train = tokenize.texts_to_matrix(train_posts)
    x_test = tokenize.texts_to_matrix(test_posts)

    encoder = LabelEncoder()
    encoder.fit(train_tags)

    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    batch_size = 32
    epochs = 2

    # Build the model
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])              

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
    print('Test accuracy:', score[1])

    predict_classes = model.predict_classes(x_test, batch_size=1)
    true_classes = np.argmax(y_test,1)
    
    print(metrics.classification_report(true_classes, predict_classes, target_names = label_names))
    cmat = metrics.confusion_matrix(true_classes, predict_classes)

    print(cmat)

if __name__ == '__main__':
    verbose = True
    stripflg = (True, True, True)

    min_freq = 10
    max_freq = 0.80

    main()
