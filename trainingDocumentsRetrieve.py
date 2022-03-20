import json

def automaticTrainingDocumentRetrieve():
    return 1

def getFragmentsFromDocument(docNumber):
    file = open('./data/automatedTrainingURLs/documents/document' + str(docNumber) + ".json" )
    obj = json.load(file)
    paragraphs = []
    for paragraph in obj:
        paragraphs.append(paragraph)
    return paragraphs
