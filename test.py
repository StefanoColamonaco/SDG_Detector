from textRetriever import retrieve, retrieveTrainigTextsFor
from analysis import initialize,check_sdg
from textClassifier import generateDatasetFor, writePairsForSDG, mergeAndOrderTuples, overwritePairsForSDG, removeDuplicatesFromOrderedTuples
from trainingDocumentsRetrieve import getFragmentsFromDocument, automaticTrainingDocumentRetrieve, getInfoFromDocument
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

newAnalysis = 0 # IMPORTANT !! Please Fil keep it on zero
# if 1: Will generate a new set of documents from which to derive training data
# if 0: Will use the current set of documents in data/automatedTrainingURLs/documents/


if __name__ == "__main__":
    if( generateDataset == 1 ):
        if( recursiveGeneration == 1 ):
            initialize()
            count = automaticTrainingDocumentRetrieve(newAnalysis)
            for i in range(1,count+1):    #count+1              # per ogni documento
                fragments = getFragmentsFromDocument(i) # split del documento 
                info = getInfoFromDocument(i)                                       #TODO: set variable as global to improve performance
                print("analizzando il documento",info['document'],"...")
                for fragment in fragments:              # per ogni frammento  
                    sdgs = check_sdg(fragment, False)          # analisi del frammento e ritorno con array degli sdg validi per frammento
                    pairs = generateDatasetFor(0, [fragment])  # generazione coppie vrb_obj
                    pairs = removeDuplicatesFromOrderedTuples(mergeAndOrderTuples(pairs))
                    for i in range(1,18):                    # concatenazione nei file degli sdg relativi 
                        if (sdgs[i-1] == 1 and info['SDG_Number']==i):
                            overwritePairsForSDG(i, pairs, [])
            print("\nAUTOMATIC GENERATION ENDED SUCCESSFULLY\n")
        else:
            for sdg in range(1,18):
                trainingPositiveTexts = retrieveTrainigTextsFor(sdg, 1)
                trainingNegativeTexts = retrieveTrainigTextsFor(sdg, 0)
                allPositivePairs = generateDatasetFor(sdg, trainingPositiveTexts)
                allNegativePairs = generateDatasetFor(sdg, trainingNegativeTexts)
                positivePairs = removeDuplicatesFromOrderedTuples(mergeAndOrderTuples(allPositivePairs))
                negativePairs = removeDuplicatesFromOrderedTuples(mergeAndOrderTuples(allNegativePairs))
                writePairsForSDG(sdg, positivePairs, negativePairs)
    texts = retrieve()
    initialize()
    for text in texts:
        check_sdg(text)
        print("\n SINGLE TASK COMPLETED \n ")
    print("\n THE ANALYZES HAVE BEEN COMPLETED \n")
