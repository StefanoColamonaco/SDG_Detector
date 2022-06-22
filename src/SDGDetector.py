import stanza
import nltk
import argparse

# argument parsing
parser = argparse.ArgumentParser(description='SDG Detector is a software that checks the presence of SDG indicators in provided texts.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--model", default=2, type=int,  help="specify the model that will be used for the classification:\n 1 - model based on manually added data in /data/trainingURL\
s\n 2 - model based on pairs obtained from targets from /data/sdgs\n 3 - model based on the generated set of documents in /data/automatedTrainingURLs", metavar="M")
parser.add_argument("-fr", "--force-rebuild", action='store_true', help="rebuild the model even if already present in /models")
opt = vars(parser.parse_args())

stanza.download('en')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

from textRetriever import retrieve, retrieveTrainigTextsFor
from analysis import initialize,check_sdg
from textClassifier import generateDatasetFor, writePairsForSDG, mergeAndOrderTuples, overwritePairsForSDG, removeDuplicatesFromOrderedTuples
from trainingDocumentsRetrieve import getFragmentsFromDocument, automaticTrainingDocumentRetrieve, getInfoFromDocument

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
    #automaticTrainingDocumentRetrieve(1,0)       # To retrieve documents without pairs extrapolation
    if( generateDataset == 1 ):
        if( recursiveGeneration == 1 ):
            initialize()
            count = automaticTrainingDocumentRetrieve(newAnalysis)
            for i in range(1,count+1):    #count+1              # per ogni documento
                fragments = getFragmentsFromDocument(i) # split del documento 
                info = getInfoFromDocument(i)                                       #TODO: set variable as global to improve performance
                print("Analyzing",info['document'],"...")
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
            print("\nSTANDARD GENERATION ENDED SUCCESSFULLY\n")
    texts = retrieve()
    initialize(opt['model'], opt['force_rebuild'])
    for text in texts:
        check_sdg(text)
        print("\n SINGLE TASK COMPLETED \n ")
    print("\n THE ANALYZES HAVE BEEN COMPLETED \n")
