import sys
import os, nltk, re, random, time
from nltk.parse import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn
from nltk.metrics import ConfusionMatrix
import json
import pickle
import re
import random
from math import floor, sqrt
from sklearn.model_selection import KFold
import numpy as np

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
        negdata = {}
        for goal in range(1, 18):
            dataset[goal] = []
            negdata[goal] = []
        file_register = open('../data/automatedTrainingURLs/documentsRegister.json')
        register = json.load(file_register)
        file_register.close()
        for regent in register:
            file_document = open('../data/automatedTrainingURLs/documents/' + regent['document'] + '.json')
            document = json.load(file_document)
            file_document.close()
            for i in range(len(document)):
                document[i] = " ".join(filter(lambda x:x[0]!='#', document[i].split()))
            document_text = " ".join(document)
            goal = regent['SDG_Number']
            dataset[goal].append((document_text, True))
        file_register = open('../data/automatedTrainingURLs/negDocumentsRegister.json')
        register = json.load(file_register)
        file_register.close()
        for regent in register:
            file_document = open('../data/automatedTrainingURLs/negDocuments/' + regent['document'] + '.json')
            document = json.load(file_document)
            file_document.close()
            for i in range(len(document)):
                document[i] = " ".join(filter(lambda x:x[0]!='#', document[i].split()))
            document_text = " ".join(document)
            goal = regent['SDG_Number']
            negdata[goal].append((document_text, False))
        for goal in range(1, 18): # avoiding unbalanced dataset through undersampling
            if len(negdata[goal]) > len(dataset[goal]):
                random.shuffle(negdata[goal])
                negdata[goal] = negdata[goal][:len(dataset[goal])]
            elif len(dataset[goal]) > len(negdata[goal]):
                random.shuffle(dataset[goal])
                dataset[goal] = dataset[goal][:len(negdata[goal])]
            dataset[goal] += negdata[goal]
            
# creating feature extractor based on verb-object pair overlap
def feature_extractor(goal, text):
    features = {} # features
    fc = 0
    pairs = []
    try:
        pairs = vrbobj_pairs(text)
    except:
        pass
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

# multiple paragraph extractor
def mp_extractor(goal, dataset):
    maxl = 1000
    featuresets = []
    for (e, g) in dataset[goal]:
        flist = {}
        for el in [e[i:i+maxl] for i in range(0, len(e), maxl)]:
            fpar = feature_extractor(goal, el)
            if len(flist) == 0:
                flist = fpar
            else:
                for ft in flist.keys():
                    flist[ft] = flist[ft] or fpar[ft]
        featuresets.append((flist, g))
    return featuresets

# training classifier
def train_model(training_option):
    print("training the classifiers for the option " + str(training_option) + "...")
    for goal in sdgir.keys():
        print("training the classifier for the goal " + str(goal) + "...")
        featuresets = mp_extractor(goal, dataset)
        train_set = featuresets
        classifier[goal] = nltk.NaiveBayesClassifier.train(train_set)
        print("the classifier for the goal " + str(goal) + " finished")
    filename = "../models/model" + ("00" + str(training_option))[-2:] + ".pickle"
    print("the classifiers for the option " + str(training_option) + " trained")
    pickle.dump(classifier, open(filename, 'wb'))

# testing classifier
def test_model(training_option):
    preload()
    load_data(training_option)
    print("training the classifiers for the option " + str(training_option) + "...")
    tmplst = [1, 8]
    for goal in tmplst: # sdgir.keys():
        print('Goal %d' % int(goal))
        featuresets = np.array(mp_extractor(goal, dataset))
        kfold = KFold(len(featuresets), random_state=1, shuffle=True)
        TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
        for train, test in kfold.split(featuresets):
            # print('iteration')
            test_set = featuresets[test]
            train_set = featuresets[train]
            # print('Train set size: %d' % len(train_set))
            # print('Test set size: %d' % len(test_set))
            classifier[goal] = nltk.NaiveBayesClassifier.train(train_set)
            ref = []
            tagged = []
            for t in test_set:
                tag = classifier[goal].classify(t[0])
                tagged.append(tag)
                ref.append(t[1])
                if tag == t[1]:
                    if tag == 1:
                        TP += 1.0
                    else:
                        TN += 1.0
                else:
                    if tag == 1:
                        FP += 1.0
                    else:
                        FN += 1.0
        ref = [True] * int(TP) + [False] * int(FP) + [True] * int(FN) + [False] * int(TN)
        tag = [True] * int(TP) + [True] * int(FP) + [False] * int(FN) + [False] * int(TN)
        cm = ConfusionMatrix(ref, tag)
        P = TP + FN
        N = TN + FP
        acc = (TP + TN) / (P + N)
        ba = ((TP / P) + (TN / N)) / 2
        try:
            kappa = (2 * (TP * TN - FP * FN)) / ((TP + FP) * (TN + FP) + (TP + FN) * (TN + FN))
        except ZeroDivisionError:
            kappa = 0
        try:
            mcc = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        except ZeroDivisionError:
            mcc = 0     
        print(cm)
        print('TP = %d' % TP)
        print('TN = %d' % TN)
        print('FP = %d' % FP)
        print('FN = %d' % FN)
        print('ACC = %f' % acc)
        print('BA = %f' % ba)
        print('kappa = %f' % kappa)
        print('MCC = %f' % mcc)
        
if __name__ == "__main__":
    training_option = int(sys.argv[1])
    testing_option = int(sys.argv[2])
    if not 1 <= training_option <= 3:
        print('This training option doesn\'t exist.')
        exit()
    if not 1 <= testing_option <= 2:
        print('This testing option doesn\'t exist.')
        exit()
    if testing_option == 1:
        build_model(training_option)
    else:
        test_model(training_option)
