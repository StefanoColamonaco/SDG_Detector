from textRetriever import retrieve, retrieveTrainigTextsFor
from analysis import initialize,check_sdg
from textClassifier import generateDatasetFor, writePairsForSDG, orderTuples
from automaticTrainingDocumentRetrieve import trainingDocumentsRetrieve
import stanza

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
stanza.download('en')

generateDataset = 0
recursiveGeneration = 0

if __name__ == "__main__":
    if(generateDataset == 1):
        if(recursiveGeneration == 1):
            count = automaticTrainingDocumentRetrieve()
            for i in range(1,count+1): #per ogni documento
                print("deb")
                # split del documento
                # per ogni frammento
                    # analisi del frammento e ritorno con array degli sdg validi per frammento
                    # generazione coppie vrb_obj
                    # concatenazione nei file degli sdg relativi

        else:
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
