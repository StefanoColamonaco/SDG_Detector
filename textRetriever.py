from bs4 import BeautifulSoup
import requests
import json

def fillURLs():
    with open('urlsList.json') as f:
        return json.load(f)

def file(url):
    #print("is a file", url)
    response = requests.get(str(url))
    text = response.text
    return text
    #print(text)
    
def site(url):
    #print("is a site", url)
    response = requests.get(str(url))
    html = response.text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    return text
    #print(text)

options = {
    "F": file,
    "S": site
}

def retrieve():
    elements = fillURLs()
    for element in elements:
        if( (element["type"] == "F" or element["type"] == "S") and  element["url"] != ""):
            return options[element["type"]](element["url"])
        else:
            print("Error during URL list analysis")
