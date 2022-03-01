## importing libraries
import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn

# loading goals and targets
# goal regex: Goal ([0-9]+): ([a-zA-Z0-9-,.:! ]+) /// g1 = goal number /// g2 = goal text
# target regex: [0-9]+.[0-9]+: ([a-zA-Z0-9-,.:! ]+) /// g1 = target text
sdgir = dict() # SDG info raw list
classifier = {} # dictionary of classifiers goal(key)->classifier(entry)
tpairs = {} # the storage of verb-object pairs for targets 
tdict = {} # the storage of verb-object pairs for sentences in text
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

def initialize():
    tdict.clear()
    preload()
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

def vrbobj_pairs(text):
    sent = nltk.word_tokenize(text)
    parse, = dep_parser.parse(sent)
    ans = []
    for governor, dep, dependent in parse.triples():
        if dep == 'obj':
            ans.append((governor[0], dependent[0]))
    return ans

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
                fc += 1
                features['vrbobj_pair_overlap'] = fc
    return features

# defining and training classifier
def init_classifiers():
    # initialization of the storage of verb-object pairs for targets
    tpairs.clear()
    for goal in sdgir.keys():
        tpairs[goal] = []
        for target in sdgir[goal][1]:
            tpairs[goal] += vrbobj_pairs("We want to " + target.lower())
    # defining classifier
    labeled_sent = [("We want to " + target.lower(), goal) for goal in sdgir.keys() for target in sdgir[goal][1]]
    random.shuffle(labeled_sent)
    tdict.clear()
    for goal in sdgir.keys():
        featuresets = [(feature_extractor(goal, e), g == goal) for (e, g) in labeled_sent]
        print('Feature sets generated for goal {}'.format(goal))
        train_set = featuresets[:70]
        classifier[goal] = nltk.NaiveBayesClassifier.train(train_set)

def check_sdg(text):   
    tdict.clear() 
    for goal in sdgir.keys():
        ans = classifier[goal].classify(feature_extractor(goal, text))
        if ans:
            print("{}: {}".format(goal, sdgir[goal][0]))
