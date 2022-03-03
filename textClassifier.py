## importing libraries
import stanza
#from stanza.server import CoreNLPClient
import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse,constituency')

#def vrbobj_pairs_prev(text):
#    sent = nltk.word_tokenize(text)
#    parse, = dep_parser.parse(sent)
#    ans = []
#    for governor, dep, dependent in parse.triples():
#        if dep == 'obj':
#            ans.append((governor[0], dependent[0]))
#    return ans

def vrbobj_pairs(text):
    doc = nlp(text)
    for sentence in doc.sentences:
        pairs = extrapolatePairs(sentence.words)
        #print(sentence.text)
        return pairs

def extrapolatePairs(words):
    pairs = []
    nouns = getNouns(words)
    for noun in nouns:
        verb = goBackToVerb(noun, words)
        if(verb != -1 and validate(verb.lemma, noun.lemma)):
            pairs.append((verb.lemma, noun.lemma,getWeightFor(verb.lemma,noun.lemma)))
    return pairs
    #print(pairs)

def goBackToVerb(word, words):
    while word.deprel != "root":
        word = words[word.head-1]
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
    return 1

def getWeightFor(verb,noun):
    return 1


