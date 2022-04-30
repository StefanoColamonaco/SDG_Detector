import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn
import json
import pickle

from textClassifier import vrbobj_pairs
from modelBuilder import initialize

# loading goals and targets
# goal regex: Goal ([0-9]+): ([a-zA-Z0-9-,.:! ]+) /// g1 = goal number /// g2 = goal text
# target regex: [0-9]+.[0-9]+: ([a-zA-Z0-9-,.:! ]+) /// g1 = target text
sdgdesc = dict() # SDG info
classifier = {} # dictionary of classifiers goal(key)->classifier(entry)
tpairs = dict() # the storage of verb-object pairs for targets 
tdict = {} # the storage of verb-object pairs for sentences in text

def initialize(data_opt=2):
    """
    Loads data and initializes the classifiers 
    Args:
        data_opt: an integer that specifies the source of data, ../data/trainingURLs if 1, ../data/sdgs if 2 and ../data/automatedTrainingURLs if 3
    """
    global classifier
    preload()
    try:
        filename = "../models/model" + ("00" + str(data_opt))[-2:] + ".pickle"
        classifier = pickle.load(open(filename, 'rb'))
        print("model found and loaded")
    except:
        print("required classifiers not found in models, proceeding to training...")
        classifier = initialize(data_opt)
    print("\n INITIALIZATION COMPLETED \n")
            
def preload():
    for entry in os.listdir('../data/sdgs'):
        file = open('../data/sdgs/' + entry)
        line = file.readline()
        gm = re.match(r'Goal ([0-9]+): ([^\n]+)', line)
        goal = int(gm.group(1))
        sdgdesc[goal] = gm.group(2)
        file.close()
    for entry in os.listdir('../data/dataset'):
        file = open('../data/dataset/' + entry)
        goal = int(entry[0:2])
        line = file.readline()
        tpairs[goal] = []
        while line:
            tpairs[goal].append((line.split()[0], line.split()[1]))
            line = file.readline()
        file.close()

# creating feature extractor based on verb-object pair overlap
def feature_extractor(goal, text):
    features = {} # features
    fc = 0
    pairs = []
    if text in tdict.keys():
        pairs = tdict[text]
    else:
        tdict[text] = pairs = vrbobj_pairs(text)
    for target in tpairs[goal]:
        features['contains(%s)' % str(target)] = False
        for p in pairs:
            vflag, oflag = False, False
            for ss in wn.synsets(target[0]):
                if p[0] in ss.lemma_names():
                    vflag = True
                    break
            if not vflag:
                continue
            for ss in wn.synsets(target[1]):
                if p[1] in ss.lemma_names():
                    oflag = True
                    break
            if vflag and oflag:
                features['contains(%s)' % str(target)] = True
                break
    return features

def check_sdg(text, output=True):
    """
    Checks the presence of SDGs in provided text 
    Args:
        text: a string which contains the text to be analysed
    Returns:
        res: a list of booleans representing the presence of goals
    """
    res = [False for i in range(17)]
    for goal in sdgdesc.keys():
        ans = classifier[goal].classify(feature_extractor(goal, text))
        if ans:
            res[goal - 1] = True
    if(output == True):
        for goal in range(1, 18):
            if res[goal - 1]:
                print("[\033[92m\u2713\033[0m] {}: {}".format(goal, sdgdesc[goal]))
            else:
                print("[\033[91m\u2717\033[0m] {}: {}".format(goal, sdgdesc[goal]))
    return res
