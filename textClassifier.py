## importing libraries
import os, nltk, re, random

# loading goals and targets
# goal regex: Goal ([0-9]+): ([a-zA-Z0-9-,.:! ]+) /// g1 = goal number /// g2 = goal text
# target regex: [0-9]+.[0-9]+: ([a-zA-Z0-9-,.:! ]+) /// g1 = target text
sdgir = dict() # SDG info raw list
classifier = {} # dictionary of classifiers goal(key)->classifier(entry)

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

def vbnp_pairs(text):
    grammar = r"""
    PP: {<IN><DT|JJ|NN.*>+}
    NP: {<DT|JJ|NN.*|CD>+}
    VP: {<RB>*<VB.*>(<TO|VB.*>|<NP|PP|,>+)*}
    """
    cp = nltk.RegexpParser(grammar)
    sent = []
    try:
        sent = nltk.pos_tag(nltk.word_tokenize(text))
    except:
        return []
    tree = cp.parse(sent)
    ans = []
    for subtree in tree.subtrees():
        if subtree.label() == 'VP':
            current_vb = None
            last = ""
            for st in subtree:
                if type(st) is nltk.tree.Tree and st.label() == 'NP' and current_vb:
                    np = ""
                    for leave in st.leaves():
                        np += leave[0] + " "
                    ans.append((current_vb, np[:-1]))
                    current_vb = None
                elif type(st) is tuple and st[1].startswith('VB'):
                    current_vb = st[0]
                last = st[0]
    return ans

# creating feature extractor based on vbnp pair overlap
def feature_extractor(goal, text):
    features = {} # features
    fc = 0
    pairs = vbnp_pairs(text)
    for target in sdgir[goal][1]:
        tpairs = vbnp_pairs("We want to " + target.lower())
        for tp in tpairs:
            if tp in pairs:
                fc += 1
                break
    features['vbnp_pair_overlap'] = fc
    return features

# defining classifier
def init_classifiers():
    labeled_sent = [("We want to " + target.lower(), goal) for goal in sdgir.keys() for target in sdgir[goal][1]]
    random.shuffle(labeled_sent)
    for goal in sdgir.keys():
        featuresets = [(feature_extractor(goal, e), g == goal) for (e, g) in labeled_sent]
        print('Feature sets generated for goal {}'.format(goal))
        train_set = featuresets[:100]
        classifier[goal] = nltk.NaiveBayesClassifier.train(train_set)

def check_sdg(text):
    preload()
    init_classifiers()
    for goal in sdgir.keys():
        ans = classifier[goal].classify(feature_extractor(goal, text))
        if ans:
            print("{}: {}".format(goal, sdgir[goal][0]))
