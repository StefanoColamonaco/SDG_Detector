from bs4 import BeautifulSoup
import requests
import json

def fillURLs():
    with open('urlsList.json') as f:
        return json.load(f)

def fillTrainingURLs(num):
    stringnum = ""
    if(num < 10):
        stringnum = "0"
    stringnum = stringnum + str(num) 
    with open('./data/trainingURLs/'+stringnum+'urls.json') as f:
        return json.load(f)

def file(url):
    #print("is a file", url)
    response = requests.get(str(url))
    text = response.text
    #print(text)
    return text
    
def site(url):
    #print("is a site", url)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(str(url),headers=headers)
    html = response.text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    #print(text)
    return text

options = {
    "F": file,
    "S": site
}

def retrieve():
    elements = fillURLs()
    texts = []
    for element in elements:
        if( (element["type"] == "F" or element["type"] == "S") and  element["url"] != ""):
            texts.append(options[element["type"]](element["url"]))
        else:
            print("Error during URL list analysis")
    return texts

def retrieveTrainigTextsFor(sdgNum, isPositive):
    elements = fillTrainingURLs(sdgNum)
    texts = []
    for element in elements:
        if( (element["type"] == "F" or element["type"] == "S") and  element["url"] != ""):
            texts.append(options[element["type"]](element["url"]))
        else:
            if(isPositive == 1 and element["type"] == "T-positive"):
                texts.append(element["text"])
            if(isPositive == 0 and element["type"] == "T-negative"):
                texts.append(element["text"])
    return texts
