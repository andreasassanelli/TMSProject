#Text Mining and Search - Final Project
#
#Students:
# Nicoli,     Paolo       833311
# Sassanelli, Andrea      123123123
#
#Chosen excercise dataset: http://qwone.com/~jason/20Newsgroups/

#Load libraries
import nltk         #NLTK - Commonly used NLP library

#import components
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

#NLTK package installer
#nltk.download()

#Define constants (file locations, etc)
datasetRootDir = "C:\\usenet\\bydate\\"
testSetRootDir = datasetRootDir + "test\\"
trainSetRootDir = datasetRootDir + "train\\"

def importDataSet(datasetLocation):
    #TODO: Group selection? Treat each group as a separate corpus?
    groupName = "alt.atheism"

    #Read the specified corpus using NLTK's PlaintextCorpusReader for multiple files in directory
    corpusDir = datasetLocation + groupName                     #Append the group name to the directory path
    corpus = PlaintextCorpusReader(corpusDir, ".*\.txt",encoding='latin1')

    #Accessing the name of the files of the corpus
    files=corpus.fileids()

    for f in files:
        print (f)

    #Hang on...
    input()
    
    #Accessing all the text of the corpus
    all_text=corpus.raw()

    print (all_text)

    #Hang on...
    input()

    #Accessing all the text for one of the files
    #news1_text=corpus.raw('news1.txtprint news1_text

#ENTRY POINT
def main():
    print("Entry Point")
    importDataSet(trainSetRootDir)
    
if __name__ == '__main__':
    main()