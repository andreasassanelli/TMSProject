#Text Mining and Search - Final Project
#
#Students:
# Nicoli,     Paolo       833311
# Sassanelli, Andrea      835119
#
#Chosen excercise dataset: http://qwone.com/~jason/20Newsgroups/

#Load libraries
import os
import nltk
import re
from time import time
import random
import json

#import components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
import numpy as np

# 20newsgroup preprocessing functions
#####################################
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
#####################################

DEBUG = False
random.seed(1)

nltk.download("punkt")
pstemmer = PorterStemmer()

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

# Pre-processing function
#####################################

def preprocDocument(document):
    """
    Pre-process document by applying tokenization, case-folding, stopwords removal and stemming
    Outputs the list of terms for the input document
    """
    # tokenize
    termlist = word_tokenize(document)

    # drop non alphabetic elements and stopwords, convert to lowercase
    termlist = [elem.lower() for elem in termlist if elem.isalpha() and elem not in stop_words]

    # stemming
    termlist = [pstemmer.stem(w) for w in termlist]

    return termlist

# Dataset import function
#####################################

def importDataSet(datasetLocation, strip_flags = (True, True, True), func = lambda x: x , class_list = None, verbose = False):
    """
    Load dataset from specified location assuming each class has its own subfolder containing document files.
    Removes header/quote/footer from each document according to boolean flags in <strip_flags> tuple.
    Applies optional pre-processing function <func> to each document individually.
    Optionally loads only classes from subfolders names specified in <class_list>
    Returns list of [docID, class, data], one for each document in dataset
    """

    if verbose: print(datasetLocation)

    # prepare empty list to return
    dataset = list()

    # for each class in sample
    for groupName in os.listdir(datasetLocation):

        # keep only classes specified in class_list if not null
        if class_list is not None and groupName not in class_list: continue

        # Append the group name to the directory path
        corpusDir = datasetLocation + groupName

        # Get list of documents files
        fileList = os.listdir(corpusDir)

        if verbose: print("%s: %s" % (groupName,len(fileList)))

        # for each file in folder
        for docID in fileList:

            # load document as string
            with open( corpusDir + '/' + docID) as fin:
                data = fin.read()

            if DEBUG:
                print("vanilla")
                print(data)
                print("-----------------------------------------------------------------------------------")

            # apply pre-processing filtering according to <strips_flags>
            if strip_flags[0]: data = strip_newsgroup_header(data)
            if strip_flags[1]: data = strip_newsgroup_quoting(data)
            if strip_flags[2]: data = strip_newsgroup_footer(data)

            if DEBUG:
                print("strip")
                print(data)
                print("---------------------------------------------------------------------------------")

            # apply pre-processing function
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

# utility function for calculating accuracy given confusion matrix
def accmat(mat):
    m = np.matrix(mat)
    return float(m.trace())/float(m.sum())

# utility function for calculating representation sparsity from Document-Term matrix
def sparsity(mat):
    tot = len(mat) * len(mat[0])
    vals = sum([sum([x == 0 for x in r]) for r in mat])
    return float(vals) / tot

# Main Function
#####################################

def main(basepath, dflims, flags, cls_dict = { 'naive' : {'Constr' : MultinomialNB , 'Params' : { } } }  , class_list = None ):
    """Loads train and test datasets form basepath directory,
    Applies pre-processing according to parameters
    - <dflims> tuple for DocumentFrequency limits
    - <flags> for head/quote/foot filtering
    Builds relative TfIDF matrix representation
    Trains and evaluates each model specified in <cls_dict>
    {<name> : { const: <model constructor>, params : <parameter dictiornary> } }
    Outputs the results in json file
    """
    print("Entry Point")

    testSetRootDir = basepath + "20news-bydate-test/"
    trainSetRootDir = basepath + "20news-bydate-train/"

    data_dict = dict()

    # dataset loading
    t0 = time()
    raw_train = importDataSet(trainSetRootDir, strip_flags=flags, func=preprocDocument, class_list=class_list ,verbose=verbose)
    print("training dataset loaded in %d seconds" % (time() -t0) )
    random.shuffle(raw_train)

    t1 = time()
    raw_test = importDataSet(testSetRootDir, strip_flags=flags, func=preprocDocument, class_list=class_list, verbose=verbose)
    print("test dataset loaded in %d seconds" % (time() -t1) )
    random.shuffle(raw_test)

    # collate corpus
    train_corpus = [ ' '.join(x[2]) for x in raw_train]
    test_corpus = [ ' '.join(x[2]) for x in raw_test]

    # extract labels
    train_labels = [x[1] for x in raw_train]
    test_labels = [x[1] for x in raw_test]

    # instantiate representation objects
    count_vect = CountVectorizer(min_df=dflims[0], max_df=dflims[1])
    tfidf_trans = TfidfTransformer()

    # build Document-Term matrix
    X_train = count_vect.fit_transform(train_corpus)
    sp = round(sparsity(X_train.todense().tolist()), 3)
    print("DFlim sparsity: %s" % sp)

    data_dict['Sparsity'] = sp
    data_dict['Shape'] = X_train.shape

    # compute Tf-Idf weights
    X_train_tfidf = tfidf_trans.fit_transform(X_train)

    # apply to test sample
    X_test = count_vect.transform(test_corpus)
    X_test_tfidf = tfidf_trans.transform(X_test)

    # Classification loop
    #######################################

    # prepare output dictionary
    res_dict = dict()

    # for each specified model
    for cls in cls_dict.keys():

        random.seed(1)

        res_dict[cls] = dict()

        # instatiate model with relative parameters
        cls_class = cls_dict[cls]['Constr']( **(cls_dict[cls]['Params']))

        # train
        t0 = time()
        cls_class.fit(X_train_tfidf, train_labels)
        res_dict[cls]['Train_time'] = round(time()-t0, 2)

        # evaluate
        t0 = time()
        y_pred = cls_class.predict(X_test_tfidf)
        res_dict[cls]['Pred_time'] = round(time()-t0, 2)

        # store results
        res_dict[cls]['CMat'] = metrics.confusion_matrix(test_labels, y_pred).tolist()
        res_dict[cls]['Acc'] = accmat(res_dict[cls]['CMat'])

    return (data_dict,res_dict)


if __name__ == '__main__':
    verbose = False

    # specifiy path
    datasetRootDir = "../datasets/20news-bydate/"

    # specify model schedule
    schedule = {
        "NaiveBayes" : {
            "Constr" : MultinomialNB,
            "Params" : {}
        },
        "SGDClassifier": {
            "Constr" : SGDClassifier,
            "Params" : {'random_state' : 1}
        },
        "KNN" : {
            "Constr": KNeighborsClassifier,
            "Params": {}
        },
        "MLP" : {
            "Constr": MLPClassifier,
            "Params": {'random_state' : 1}
        }
    }

    min_freq = 10
    max_freq = 0.75

    # pre-processing sweep
    for head in [True,False]:
        for quote in [True,False]:
            for sign in [True,False]:
                stripflg = (head, quote, sign)

                # create unique file fingerprint using config parameters
                fingerprint = ''.join(map(lambda x: str(int(x)),stripflg))

                fingerprint += "_%s_%s" % (min_freq, max_freq)
                fingerprint += 'L'

                print(fingerprint)

                # core routine
                output = main(basepath=datasetRootDir, dflims=(min_freq,max_freq), flags=stripflg, cls_dict=schedule,
                              class_list=['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'])

                # write output
                with open('../reports/redump_%s.json' % fingerprint, 'w' ) as fout:
                    json.dump(output, fout)

                for k in output[1]:
                    print("%s:\t%s\t%s" % (k, round(output[1][k]['Acc'],3), output[1][k]["Train_time"]))
