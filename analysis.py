import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn

from textClassifier import vrbobj_pairs

# loading goals and targets
# goal regex: Goal ([0-9]+): ([a-zA-Z0-9-,.:! ]+) /// g1 = goal number /// g2 = goal text
# target regex: [0-9]+.[0-9]+: ([a-zA-Z0-9-,.:! ]+) /// g1 = target text
sdgir = dict() # SDG info raw list
classifier = {} # dictionary of classifiers goal(key)->classifier(entry)
tpairs = dict() # the storage of verb-object pairs for targets 
tdict = {} # the storage of verb-object pairs for sentences in text

def initialize():
    preload()
    init_classifiers()
    #printGeneratedCouples()
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
    labeled_sent = [("We want to " + target.lower(), goal) for goal in sdgir.keys() for target in sdgir[goal][1]]
    random.shuffle(labeled_sent)
    tdict.clear()
    print("generating feature sets...")
    for goal in sdgir.keys():
        featuresets = [(feature_extractor(goal, e), g == goal) for (e, g) in labeled_sent]
        print('Feature sets generated for goal {}'.format(goal))
        train_set = featuresets
        classifier[goal] = nltk.NaiveBayesClassifier.train(train_set)

def check_sdg(text):
    for goal in sdgir.keys():
        ans = classifier[goal].classify(feature_extractor(goal, text))
        if ans:
            print("+ {}: {}".format(goal, sdgir[goal][0]))
        else:
            print("- {}: {}".format(goal, sdgir[goal][0]))
