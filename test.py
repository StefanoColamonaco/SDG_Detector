from textRetriever import retrieve
from textClassifier import check_sdg

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

if __name__ == "__main__":
    check_sdg(retrieve())
