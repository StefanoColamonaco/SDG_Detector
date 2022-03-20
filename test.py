from textRetriever import retrieve, retrieveTrainigTextsFor
from analysis import initialize,check_sdg
from textClassifier import generateDatasetFor, writePairsForSDG, orderTuples
from automaticTrainingDocumentRetrieve import trainingDocumentsRetrieve # getFragmentsFromDocument
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
                # split del documento fragments = getFragmentsFromDocument(i)
                # per ogni frammento  for fragment in fragments:
                    # analisi del frammento e ritorno con array degli sdg validi per frammento sdgs = check_sdg(fragment)
                    # generazione coppie vrb_obj   pairs = generateDatasetFor(0, [fragment]):
                    # concatenazione nei file degli sdg relativi for sdg in sdgs:
                    # if (sdg == 1):
                    # overwritePairsForSDG(sdg, pairs, [])


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
