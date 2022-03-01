from textRetriever import retrieve
from textClassifier import check_sdg
from textClassifier import initialize

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

if __name__ == "__main__":
    texts = retrieve()
    initialize()
    for text in texts:
        check_sdg(text)
        print("\n SINGLE TASK COMPLETED \n ")
    print("\n THE ANALYZES HAVE BEEN COMPLETED \n")
