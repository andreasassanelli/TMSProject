#Text Mining and Search - Final Project
#
#Students:
# Nicoli,     Paolo       833311
# Sassanelli, Andrea      123123123
#
#Chosen excercise dataset: http://qwone.com/~jason/20Newsgroups/

#Load libraries
import os
import nltk
import re
from time import time
import random
import collections
import string
from nltk.classify.scikitlearn import SklearnClassifier

#import components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from matplotlib import pyplot as plt
# Additional
from sklearn.naive_bayes import BernoulliNB
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.metrics import (precision, recall)

class IgnoreHeadingCorpusView(StreamBackedCorpusView):
    def __init__(self, *args, **kwargs):
        StreamBackedCorpusView.__init__(self, *args, **kwargs)
        # open self._stream
        self._open()
        # skip the heading block
        self.read_block(self._stream)
        # reset the start position to the current position in the stream
        self._filepos = [self._stream.tell()]
        self.close()

class IgnoreHeadingCorpusReader(CategorizedPlaintextCorpusReader):
    CorpusView = IgnoreHeadingCorpusView


# 20newsgroup preprocessing functions
# source: scikit-learn/sklearn/datasets/twenty_newsgroups.py
_HEADER_RE = re.compile(r'^[A-Za-z0-9-]*:')
_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')

def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    Additionally, drop lines leading with a header-like pattern, ie "Key-Text:"
    """
    _before, _blankline, after = text.partition('\n\n')
    after_clean = [line for line in after.split('\n')
                  if not _HEADER_RE.search(line)]
    return '\n'.join(after_clean)

def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.
    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


# Define constants (file locations, etc)
DEBUG = False
random.seed(1)

datasetRootDir = "c:/datasets/20news-bydate/"
testSetDir = "20news-bydate-test/"
trainSetDir = "20news-bydate-train/"
testSetFixedDir = "20news-bydate-test-pre/"
trainSetFixedDir = "20news-bydate-train-pre/"
testSetRootDir = datasetRootDir + "20news-bydate-test"
trainSetRootDir = datasetRootDir + "20news-bydate-train"

nltk.download("words")

nltk.download("punkt")

nltk.download('wordnet')

pstemmer = PorterStemmer()

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

def preprocDocument(document):
    """
    Pre-process document by applying tokenization, stopwords removal and stemming
    """
    # tokenize
    termlist = word_tokenize(document)

    # drop non alphabetic elements and stopwords, convert to lowercase
    termlist = [elem.lower() for elem in termlist if elem.isalpha() and elem not in stop_words]

    lem = WordNetLemmatizer()
    #termlist = map(lem.lemmatize, termlist)

    # only english
    #termlist = [word.lower() for word in words.words() if word in termlist]

    
#
#stem = PorterStemmer()

    # stemming
    termlist = [pstemmer.stem(w) for w in termlist]

    return termlist

def preprocessDataSet(datasetRoot, SourceDIR, DestDIR, strip_flags = (True, True, True), func = lambda x: x , verbose = False):
    """
    Load dataset from specified location assuming each class has its own subfolder containing document files.
    Removes header/quote/footer from each document according to boolean flags in <strip_flags> tuple.
    Applies optional pre-processing function <func> to each document individually.
    Returns list of [docID, class, data], one for each document in dataset
    """

    datasetLocation = datasetRoot + SourceDIR

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

            #train_corpus = [ ' '.join(x[2]) for x in raw_train]
            #test_corpus = [ ' '.join(x[2]) for x in raw_test]

            with open( datasetRoot + DestDIR + groupName + '/' + docID, "w") as fout:
                fout.write(' '.join(data))
                fout.close()
            # append resulting tuple to output dataset
            #dataset.append([docID, groupName, data])
            #dataset.append([data, groupName])

            if DEBUG: break
        if DEBUG: break

    return dataset

#Find Features
def find_features(document, word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#ENTRY POINT
def main():
    print("Entry Point")
    
    #preprocessDataSet(datasetRootDir, trainSetDir, trainSetFixedDir, strip_flags=stripflg, func=preprocDocument, verbose=verbose)
    #preprocessDataSet(datasetRootDir, testSetDir, testSetFixedDir, strip_flags=stripflg, func=preprocDocument, verbose=verbose)
    
    # Load Corpora  IgnoreHeadingCorpusReader
    usenet_train = CategorizedPlaintextCorpusReader(trainSetRootDir + "-pre/", r'.*', cat_pattern=r'((\w+[.]?)*)/*', encoding="ISO-8859-1")
    usenet_test = CategorizedPlaintextCorpusReader(testSetRootDir + "-pre/", r'.*', cat_pattern=r'((\w+[.]?)*)/*', encoding="ISO-8859-1")

    #print(usenet_train.words())                                                                                                                                                                            
    #print(usenet_train.categories())
    #print(usenet_train.fileids())
   
    train_docs = [(list(usenet_train.words(fileid)), category)
                 for category in usenet_train.categories()
                 for fileid in usenet_train.fileids(category)]
    

    #train_docs = [([w for w in usenet_train.words(fileid) if w not in stop_words], category)
    #             for category in usenet_train.categories()
    #             for fileid in usenet_train.fileids(category)]
    
    #print(train_docs)

    test_docs = [(list(usenet_test.words(fileid)), category)
                 for category in usenet_test.categories()
                 for fileid in usenet_test.fileids(category)]
                 
    random.shuffle(train_docs)
    random.shuffle(test_docs)

    #print(train_docs[1])

    all_words = []
    #for w in usenet_train.words():
    for w in usenet_train.words():
        all_words.append(w)

    all_words = nltk.FreqDist(all_words)
    #print(all_words.most_common(15))
    #print(all_words["stupid"])

    all_words_test = []
    for w in usenet_test.words():
        all_words_test.append(w)

    all_words = nltk.FreqDist(all_words)
    all_words_test = nltk.FreqDist(all_words_test)
    
    print(len(all_words.keys()))
    word_features = list(all_words.keys())[:2000]
    word_features_test = list(all_words_test.keys())[:2000]
    #print((find_features(usenet_train.words('alt.atheism/51130'),word_features)))
    
    training_set = [(find_features(rev, word_features), category) for (rev, category) in train_docs]
    testing_set = [(find_features(rev, word_features_test), category) for (rev, category) in test_docs]

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("NLTK Bayes Classifier accuracy percent: ",(nltk.classify.accuracy(classifier, testing_set))*100)

    #classifier.show_most_informative_features(15)

    #OUT
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testing_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
 
    for cat in usenet_test.categories():
        print (cat + ' precision:', precision(refsets[cat], testsets[cat]))
        print (cat + ' recall:', recall(refsets[cat], testsets[cat]))

    # Try More
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MultinomialNB accuracy rcent: ",nltk.classify.accuracy(MNB_classifier, testing_set)*100)



    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_set)
    print("BernoulliNB accuracy percent: ",nltk.classify.accuracy(BNB_classifier, testing_set)*100)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='lbfgs',multi_class='multinomial'))
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

    SVC_classifier = SklearnClassifier(SVC(gamma='scale'))
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)     
    


    # KERAS
    # Building KERAS model
    # Input - Layer
    model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu")
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation = "sigmoid"))
    model.summary()


if __name__ == '__main__':
    verbose = True
    stripflg = (True, True, True)

    min_freq = 0.05
    max_freq = 0.80

    main()
