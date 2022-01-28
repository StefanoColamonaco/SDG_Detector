from bs4 import BeautifulSoup
import requests

urls = []

def fillURLs():
    with open('urlsList.txt') as f:
        lines = f.readlines()
        for line in lines:
            urls.append(line)


def retrieve():
    fillURLs()
    for url in urls:
        response = requests.get(str(url))
        html = response.text

        soup = BeautifulSoup(html, features="html.parser")
        # print(soup.get_text())
        return soup.get_text()
