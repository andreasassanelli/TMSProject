#Text Mining and Search - Final Project
#
#Students:
# Nicoli,     Paolo       833311
# Sassanelli, Andrea      123123123
#
#Chosen excercise dataset: http://qwone.com/~jason/20Newsgroups/

#Load libraries
import os
import pandas as pd
import nltk
import re

#import components
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

#NLTK package installer
#nltk.download()

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
datasetRootDir = "../datasets/20news-bydate/"
testSetRootDir = datasetRootDir + "20news-bydate-test/"
trainSetRootDir = datasetRootDir + "20news-bydate-train/"

def importDataSet(datasetLocation, strips = (True,True,True), func = lambda x: x , verbose = False):

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

            # apply pre-processing functions according to "strips" flags
            if strips[0]: data = strip_newsgroup_header(data)
            if strips[1]: data = strip_newsgroup_quoting(data)
            if strips[2]: data = strip_newsgroup_footer(data)

            # apply additional processing if needed
            data = func(data)

            # append resulting tuple to output dataset
            dataset.append([docID, groupName, data])

    return dataset

#ENTRY POINT
def main():
    print("Entry Point")

    raw_train = importDataSet(trainSetRootDir)
    raw_test = importDataSet(testSetRootDir)

if __name__ == '__main__':
    main()