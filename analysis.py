import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn
import json

from textClassifier import vrbobj_pairs

# loading goals and targets
# goal regex: Goal ([0-9]+): ([a-zA-Z0-9-,.:! ]+) /// g1 = goal number /// g2 = goal text
# target regex: [0-9]+.[0-9]+: ([a-zA-Z0-9-,.:! ]+) /// g1 = target text
sdgir = dict() # SDG info raw list
dataset = dict() # dictionary of data useful for training and testing classifiers
classifier = {} # dictionary of classifiers goal(key)->classifier(entry)
tpairs = dict() # the storage of verb-object pairs for targets 
tdict = {} # the storage of verb-object pairs for sentences in text

def initialize(data_opt=False):
    """
    Loads data and initializes the classifiers 
    Args:
        data_opt: a boolean that specifies the source of data, ./data/trainingURLs if True, ./data/sdgs otherwise
    """
    preload()
    load_data(data_opt)
    init_classifiers()
    print("\n INITIALIZATION COMPLETED \n")
            
def preload():
    for entry in os.listdir('./data/sdgs'):
        file = open('./data/sdgs/' + entry)
        line = file.readline()
        gm = re.match(r'Goal ([0-9]+): ([^\n]+)', line)
        goal = int(gm.group(1))
        sdgir[goal] = (gm.group(2), [])
        file.readline()
        while line:
            tm = re.match(r'[0-9]+.[0-9]+: ([^\n]+)', line)
            if tm:
                sdgir[goal][1].append(tm.group(1))
            line = file.readline()
        file.close()
    for entry in os.listdir('./data/dataset'):
        file = open('./data/dataset/' + entry)
        goal = int(entry[0:2])
        line = file.readline()
        tpairs[goal] = []
        while line:
            tpairs[goal].append((line.split()[0], line.split()[1]))
            line = file.readline()
        file.close()

def load_data(opt):
    if opt:
        for dirent in os.listdir('./data/trainingURLs'):
            file = open('./data/trainingURLs/' + dirent)
            goal = int(dirent[0:2])
            data = json.load(file)
            dataset[goal] = []
            for entry in data:
                dataset[goal].append((entry["text"], entry["type"] == "T-positive"))
            random.shuffle(dataset[goal])
            file.close()
    else:
        for goal in range(1, 18):
            dataset[goal] = []
            for gcmp in range(1, 18):
                dataset[goal] += [(entry, goal == gcmp) for entry in sdgir[gcmp][1]]
        
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

# defining and training classifier
def init_classifiers():
    # defining classifier
    tdict.clear()
    print("training the classifiers...")
    for goal in sdgir.keys():
        featuresets = [(feature_extractor(goal, e), g) for (e, g) in dataset[goal]]
        train_set = featuresets
        classifier[goal] = nltk.NaiveBayesClassifier.train(train_set)

def check_sdg(text):
    """
    Checks the presence of SDGs in provided text 
    Args:
        text: a string which contains the text to be analysed
    Returns:
        res: a list of booleans representing the presence of goals
    """
    res = [False for i in range(17)]
    for goal in sdgir.keys():
        ans = classifier[goal].classify(feature_extractor(goal, text))
        if ans:
            res[goal - 1] = True
    for goal in range(1, 18):
        if res[goal - 1]:
            print("[\033[92m\u2713\033[0m] {}: {}".format(goal, sdgir[goal][0]))
        else:
            print("[\033[91m\u2717\033[0m] {}: {}".format(goal, sdgir[goal][0]))
    return res
