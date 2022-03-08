from textRetriever import retrieve, retrieveTrainigTextsFor
from analysis import initialize,check_sdg
from textClassifier import generateDatasetFor, writePairsForSDG, orderTuples
import stanza

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
stanza.download('en')

generateDataset = 0

if __name__ == "__main__":
    if(generateDataset == 1):
        for sdg in range(1,18):
            trainingPositiveTexts = retrieveTrainigTextsFor(sdg, 1)
            trainingNegativeTexts = retrieveTrainigTextsFor(sdg, 0)
            allPositivePairs = generateDatasetFor(sdg, trainingPositiveTexts)
            allNegativePairs = generateDatasetFor(sdg, trainingNegativeTexts)
            positivePairs = orderTuples(allPositivePairs)
            negativePairs = orderTuples(allNegativePairs)
            writePairsForSDG(sdg, positivePairs, negativePairs)
    texts = retrieve()
    initialize()
    for text in texts:
        check_sdg(text)
        print("\n SINGLE TASK COMPLETED \n ")
    print("\n THE ANALYZES HAVE BEEN COMPLETED \n")
