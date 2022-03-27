from textRetriever import retrieve, retrieveTrainigTextsFor
from analysis import initialize,check_sdg
from textClassifier import generateDatasetFor, writePairsForSDG, orderTuples, overwritePairsForSDG
from trainingDocumentsRetrieve import getFragmentsFromDocument, automaticTrainingDocumentRetrieve
import stanza
import nltk

stanza.download('en')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

generateDataset = 0
# if 1: Will generate dataset
# if 0: Will use the current dataset

recursiveGeneration = 0
# if 1: Dataset generation will use recursive generation tecnique 
# if 0: Will use the first implementation of dataset generation from hand-written phrases

newAnalysis = 0    # Please Fil keep it on zero
# if 1: Will generate a new set of documents from which to derive training data
# if 0: Will use the current set of documents in data/automatedTrainingURLs/documents/


if __name__ == "__main__":
    if( generateDataset == 1 ):
        if( recursiveGeneration == 1 ):
            #initialize()
            count = automaticTrainingDocumentRetrieve(newAnalysis)
            print(count)
            for i in range(1,1):    #count+1              # per ogni documento
                fragments = getFragmentsFromDocument(i) # split del documento 
                for fragment in fragments:              # per ogni frammento  
                    sdgs = check_sdg(fragment)          # analisi del frammento e ritorno con array degli sdg validi per frammento
                    pairs = generateDatasetFor(0, [fragment])  # generazione coppie vrb_obj
                    pairs = orderTuples(pairs)
                    print(pairs)
                    for i in range(1,18):                    # concatenazione nei file degli sdg relativi 
                        if (sdgs[i-1] == 1):
                            print(i)
                            overwritePairsForSDG(i, pairs, [])
            print("automatic generation ended")
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
