import sys
import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn
import json
import pickle
import re

from textClassifier import vrbobj_pairs

# loading goals and targets
# goal regex: Goal ([0-9]+): ([a-zA-Z0-9-,.:! ]+) /// g1 = goal number /// g2 = goal text
# target regex: [0-9]+.[0-9]+: ([a-zA-Z0-9-,.:! ]+) /// g1 = target text
sdgir = dict() # SDG info raw list
dataset = dict() # dictionary of data useful for training and testing classifiers
classifier = {} # dictionary of classifiers goal(key)->classifier(entry)
tpairs = dict() # the storage of verb-object pairs for targets 

def build_model(training_option):
    """
    Loads data and initializes the classifiers 
    Args:
        data_opt: an integer that specifies the source of data, ../data/trainingURLs if 1, ../data/sdgs if 2 and ../data/automatedTrainingURLs if 3
    """
    preload()
    load_data(training_option)
    train_model(training_option)
    return classifier
            
def preload():
    for entry in os.listdir('../data/sdgs'):
        file = open('../data/sdgs/' + entry)
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
    for entry in os.listdir('../data/dataset'):
        file = open('../data/dataset/' + entry)
        goal = int(entry[0:2])
        line = file.readline()
        tpairs[goal] = []
        while line:
            tpairs[goal].append((line.split()[0], line.split()[1]))
            line = file.readline()
        file.close()

def load_data(opt):
    if opt == 1:
        for dirent in os.listdir('../data/trainingURLs'):
            file = open('../data/trainingURLs/' + dirent)
            goal = int(dirent[0:2])
            data = json.load(file)
            dataset[goal] = []
            for entry in data:
                dataset[goal].append((entry["text"], entry["type"] == "T-positive"))
                random.shuffle(dataset[goal])
            file.close()
    elif opt == 2:
        for goal in range(1, 18):
            dataset[goal] = []
            for gcmp in range(1, 18):
                dataset[goal] += [(entry, goal == gcmp) for entry in sdgir[gcmp][1]]
    elif opt == 3:
        for goal in range(1, 18):
            dataset[goal] = []
        file_register = open('../data/automatedTrainingURLs/documentsRegister.json')
        register = json.load(file_register)
        file_register.close()
        for regent in register:
            file_document = open('../data/automatedTrainingURLs/documents/' + regent['document'] + '.json')
            document = json.load(file_document)
            file_document.close()
            document_text = ' '.join(document)
            CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
            document_text = re.sub(CLEANR, '', document_text)
            goal = regent['SDG_Number']
            dataset[goal].append((document_text, True))
            # TODO: negative texts

                
# creating feature extractor based on verb-object pair overlap
def feature_extractor(goal, text):
    features = {} # features
    fc = 0
    pairs = vrbobj_pairs(text)
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

# training classifier
def train_model(training_option):
    print("training the classifiers for the option " + str(training_option) + "...")
    for goal in sdgir.keys():
        print("training the classifier for the goal " + str(goal) + "...")
        featuresets = [(feature_extractor(goal, e), g) for (e, g) in dataset[goal]]
        train_set = featuresets
        classifier[goal] = nltk.NaiveBayesClassifier.train(train_set)
        print("the classifier for the goal " + str(goal) + " finished")
    filename = "../models/model" + ("00" + str(training_option))[-2:] + ".pickle"
    print("the classifiers for the option " + str(training_option) + " trained")
    pickle.dump(classifier, open(filename, 'wb'))

if __name__ == "__main__":
    training_option = int(sys.argv[1])
    if not 1 <= training_option <= 3:
        print('This option doesn\'t exist.')
        exit()
    build_model(training_option)
