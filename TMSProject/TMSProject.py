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
import nltk         #NLTK - Commonly used NLP library

#import components
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

#NLTK package installer
#nltk.download()

#Define constants (file locations, etc)
datasetRootDir = "..\\datasets\\20news-bydate\\"
testSetRootDir = datasetRootDir + "20news-bydate-test/"
trainSetRootDir = datasetRootDir + "20news-bydate-train\\"

def importDataSet(datasetLocation):

    for groupName in os.listdir(datasetLocation):

        #Read the specified corpus using NLTK's PlaintextCorpusReader for multiple files in directory
        corpusDir = datasetLocation + groupName                     #Append the group name to the directory path
        fileList = os.listdir(corpusDir)
        corpus = PlaintextCorpusReader(corpusDir, fileList, encoding='latin1')

        #Accessing the name of the files of the corpus
        files=corpus.fileids()

        print("%s: %s" % (groupName,len(files)))

        #Hang on...
        input()

        #Accessing all the text of the corpus
        all_text=corpus.raw()

        #print(all_text)

        #Hang on...
        #input()

        #Accessing all the text for one of the files
        #news1_text=corpus.raw('news1.txtprint news1_text

#ENTRY POINT
def main():
    print("Entry Point")
    importDataSet(trainSetRootDir)
    
if __name__ == '__main__':
    main()