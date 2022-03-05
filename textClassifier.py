## importing libraries
import stanza
#from stanza.server import CoreNLPClient
import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
import json
from nltk.corpus import wordnet as wn

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse,constituency')

with open('./data/options/blacklist.json') as f:
    blObj = json.load(f)
    blacklistedVerbs = blObj["blacklistedVerbs"]
    blacklistedNouns = blObj["blacklistedNouns"]
    blacklistedCouples = blObj["blacklistedCouples"]

def orderTuples(allPairs):
    tmp = []
    for pairs in allPairs:
        tmp = tmp + pairs
    tmp.sort(key=lambda x:x[1])
    tmp.sort(key=lambda x:x[0])
    return tmp

def writePairsForSDG(sdg, pairs):
    stringnum = ""
    if(sdg < 10):
        stringnum = "0"
    stringnum = stringnum + str(sdg) 
    with open('./data/dataset/'+stringnum+'pairs.txt','w') as f:
        for tup in pairs:
            text = str(str(tup[0])+" "+str(tup[1])+" "+str(tup[2])+"\n")
            f.write(text)
        

def generateDatasetFor(sdgNum, texts):
    allPairs = []
    for text in texts:
        pairs = vrbobj_pairs(text)
        allPairs.append(pairs)
    return allPairs


def vrbobj_pairs(text):
    doc = nlp(text)
    allPairs = []
    for sentence in doc.sentences:
        pairs = extrapolatePairs(sentence.words)
        #print(sentence.text)
        allPairs = allPairs + pairs
    return allPairs

def extrapolatePairs(words):
    pairs = []
    nouns = getNouns(words)
    for noun in nouns:
        verb = goBackToVerb(noun, words)
        if(verb != -1 and validate(verb.lemma, noun.lemma)):
            pairs.append((verb.lemma, noun.lemma,getWeightFor(verb.lemma,noun.lemma)))
    return pairs

def goBackToVerb(word, words):
    while word.deprel != "root":
        word = words[word.head-1]
        #This is an extra filter, verify if necessary
        if(word.upos == "NOUN"):
            return -1
        if(word.upos == "VERB"):
            return word;
    return -1

def getNouns(words):
    toReturn = []
    for word in words:
        if(word.upos == "NOUN"):
            toReturn.append(word)
    return toReturn

def validate(verb,noun):
    if(verb in blacklistedVerbs):
        return 0
    if(noun in blacklistedNouns):
        return 0
    for couple in blacklistedCouples:
        if(couple["verb"] == verb and couple["noun"] == noun):
            print(verb,noun)
            return 0
    return 1

def getWeightFor(verb,noun):
    return 1


