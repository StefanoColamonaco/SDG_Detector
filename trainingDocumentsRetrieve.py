import json
import nltk
from googlesearch import search
from textRetriever import site 

NUMSENTENCESPERPARAGRAPH = 5
DOCSPERSDG = 1
SDGOBJECTIVES = [ "No poverty", "Zero hunger", "Good health and well-being", "Quality education", "Gender equality", "Clean water and sanitation", "Affordable and clean energy", "Decent work and economic growth", "Industry, innovation and infrastructure", "Reduced inequalities", "Sustainable cities and communities", "Responsible consumption and production", "Climate action", "Life below water", "Life on land", "Peace, justice and strong institutions", "Partnerships for the sustainable goals" ]
TRAININGCOMPANIES = ["Google", "Microsoft", "Apple", "Amazon" ]

def automaticTrainingDocumentRetrieve(newAnalysis):
    count = 0
    if(newAnalysis == 1):
        dataForRegister = []
        for objIndex in range(0,17):
            for company in TRAININGCOMPANIES:
                objective = SDGOBJECTIVES[objIndex]
                query = objective+" "+"at "+company
                urls = search(query, num_results=DOCSPERSDG)
                for url in urls:
                    text = site(url)
                    data = splitTextIntoParagraphs(text)
                    path = "./data/automatedTrainingURLs/documents/" 
                    filename = path+"document"+str(count+1)+".json"
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    dataForRegister.append({"document": "document"+str(count+1), "SDG_Number":objIndex+1,"SDG":objective,"Company":company,"site":url})

        filename = "./data/automatedTrainingURLs/documentsRegister.json" 
        with open(filename, 'w') as f:
            json.dump(dataForRegister, f, indent=2)
    else:
        filename = "./data/automatedTrainingURLs/documentsRegister.json" 
        with open(filename, 'r') as f:
            registerData = json.load(f)
            count = len(registerData)
    return count

def splitTextIntoParagraphs(text):
    data = []
    sentences = nltk.sent_tokenize(text)
    paragraph = ""
    for i in range(0,len(sentences)):
        paragraph = paragraph + sentences[i] + " "
        if((i+1)&NUMSENTENCESPERPARAGRAPH == 0):
            data.append(paragraph)
            paragraph = ""
    return data
    

def getFragmentsFromDocument(docNumber):
    file = open('./data/automatedTrainingURLs/documents/document' + str(docNumber) + ".json" )
    obj = json.load(file)
    paragraphs = []
    for paragraph in obj:
        paragraphs.append(paragraph)
    return paragraphs

def getInfoFromDocument(docNumber):
    filename = "./data/automatedTrainingURLs/documentsRegister.json" 
    with open(filename, 'r') as f:
        registerData = json.load(f)
        return registerData[docNumber-1]