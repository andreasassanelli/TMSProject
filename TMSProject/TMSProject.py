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
testSetRootDir = datasetRootDir + "20news-bydate-test/"
trainSetRootDir = datasetRootDir + "20news-bydate-train/"

nltk.download("punkt")
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

    # stemming
    termlist = [pstemmer.stem(w) for w in termlist]

    return termlist

def importDataSet(datasetLocation, strip_flags = (True, True, True), func = lambda x: x , verbose = False):
    """
    Load dataset from specified location assuming each class has its own subfolder containing document files.
    Removes header/quote/footer from each document according to boolean flags in <strip_flags> tuple.
    Applies optional pre-processing function <func> to each document individually.
    Returns list of [docID, class, data], one for each document in dataset
    """

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

#ENTRY POINT
def main():
    print("Entry Point")

    # dataset loading
    t0 = time()
    raw_train = importDataSet(trainSetRootDir, strip_flags=stripflg, func=preprocDocument, verbose=verbose)
    print("training dataset loaded in %d seconds" % (time() -t0) )
    random.shuffle(raw_train)

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

    # build Document-Term matrix with TF-IDF weights
    X_train = count_vect.fit_transform(train_corpus)
    X_train_tfidf = tfidf_trans.fit_transform(X_train)

    # Classification
    clf = MultinomialNB().fit(X_train_tfidf, train_labels)

    X_test = count_vect.transform(test_corpus)
    X_test_tfidf = tfidf_trans.transform(X_test)

    y_predict = clf.predict(X_test_tfidf)

    # Evaluation
    print(metrics.classification_report(test_labels, y_predict, target_names = label_names))
    cmat = metrics.confusion_matrix(test_labels, y_predict)

    print(cmat)

    #plt.imshow(cmat)
    #plt.yticks(range(len(label_names)), label_names)
    #plt.show()

    # Try More
    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_set)
    print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))

if __name__ == '__main__':
    verbose = True
    stripflg = (True, True, True)

    min_freq = 0.05
    max_freq = 0.80

    main()