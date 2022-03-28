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

def removeDuplicatesFromOrderedTuples(orderedPairs):
    tmp = []
    last = ("","",0)
    for pair in orderedPairs:
        if(pair != last):
            tmp.append(pair)
            last = pair
    return tmp


def mergeAndOrderTuples(allPairs):
    tmp = []
    for pair in allPairs:
        tmp = tmp + pair
    return orderTuples(tmp)

def orderTuples(pairs):
    pairs.sort(key=lambda x:x[1])
    pairs.sort(key=lambda x:x[0])
    return pairs

def writePairsForSDG(sdg, positivePairs, negativePairs): #TODO: implementare eliminazione dei duplicati
    stringnum = ""
    if(sdg < 10):
        stringnum = "0"
    stringnum = stringnum + str(sdg) 
    with open('./data/dataset/'+stringnum+'pairs.txt','w') as f:
        for tup in positivePairs:
            text = str(str(tup[0])+" "+str(tup[1])+" "+str(getWeightFor(str(tup[0]),str(tup[1]),1))+"\n")
            f.write(text)
        for tup in negativePairs:
            text = str(str(tup[0])+" "+str(tup[1])+" "+str(getWeightFor(str(tup[0]),str(tup[1]),0))+"\n")
            f.write(text)
        
def overwritePairsForSDG(sdg, positivePairs, negativePairs):
    stringnum = ""
    if(sdg < 10):
        stringnum = "0"
    stringnum = stringnum + str(sdg)
    oldPositive = []
    oldNegative = []
    with open('./data/dataset/'+stringnum+'pairs.txt','r') as f:
        line = f.readline()
        while line:
            if( line.find("-1") != -1 ):
                oldNegative.append( (line.split()[0],line.split()[1]) )
            else:
                oldPositive.append( (line.split()[0],line.split()[1]) )
            line = f.readline()
    writePairsForSDG(sdg, removeDuplicatesFromOrderedTuples(orderTuples(oldPositive + positivePairs)), removeDuplicatesFromOrderedTuples(orderTuples(oldNegative + negativePairs)))

def generateDatasetFor(sdgNum, texts):
    allPairs = []
    for text in texts:
        pairs = vrbobj_pairs(text)
        allPairs.append(pairs)
    return allPairs


def vrbobj_pairs(text):
    try:
        doc = nlp(text)
        allPairs = []
        for sentence in doc.sentences:
            pairs = extrapolatePairs(sentence.words)
            #print(sentence.text)
            allPairs = allPairs + pairs
        return allPairs
    except:
        print("error in constituency parsing")
        return []

def extrapolatePairs(words):
    pairs = []
    nouns = getNouns(words)
    for noun in nouns:
        verb = goBackToVerb(noun, words)
        if(verb != -1 and validate(verb.lemma, noun.lemma)):
            pairs.append((verb.lemma, noun.lemma,0))
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
            return 0
    return 1

def getWeightFor(verb,noun, isPositive):
    if(isPositive == 1):
        return 1
    else:
        return -1


