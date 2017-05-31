from __future__ import print_function
import sys
import midi
import os
import shutil
import glob
import cPickle as pickle
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn import svm, linear_model, discriminant_analysis, naive_bayes
import itertools
import operator
import numpy
import random

import preprocess as pre
import feature_extraction as fe

def main(argv):

    genres = ['50s', '60s', '70s', '80s', '90s'] # 'country', 'past_century'
    feature_names = ['unigram', 'bigram', 'trigram', 'interval']
    # powerset of features (2^4 = 16)
    p_set = list(itertools.chain.from_iterable(itertools.combinations(feature_names, r) for r in range(len(feature_names)+1)))


    # preprocess all midi's
    for genre in genres:
        # build list of filenames
        filenames = []
        for ext in ['.mid', '.kar']:
            for f in os.listdir('data/' + genre + '/'):
                if f.endswith(ext):
                    filenames.append(os.path.join('data/' + genre + '/', f))

        print("preprocessing " + genre + " music files...")
        for filename in filenames:
            base = os.path.basename(filename)
            name = os.path.splitext(base)
            # preprocess midi if it doesn't have a corresponding .p file
            if not os.path.exists('preprocess_pickle/' + genre + '/' + name[0] + '.p'):
                print("\tpreprocessing " + name[0] + "...", end='')
                # read the midi file
                pattern = midi.read_midifile(filename)

                # preprocess the midi file
                out = pre.preprocess(pattern)

                # pickle the output
                pickle.dump(out, open("preprocess_pickle/" + genre + "/" + name[0] + ".p", "wb"))
                print(" done.")
            # otherwise skip
            else:
                print("\tskipping " + name[0] + ".")
                continue



    # build list of filenames
    filenames = []
    for genre in genres:
            for f in os.listdir('preprocess_pickle/' + genre + '/'):
                if f.endswith('.p'):
                    filenames.append(os.path.join('preprocess_pickle/' + genre + '/', f))

    # make label vector
    labels = []
    for genre in genres:
        n = len(glob.glob(os.path.join('preprocess_pickle/' + genre + '/', '*.p')))
        labels.extend([genre] * n)

    X = numpy.array(filenames)
    y = numpy.array(labels)
    kf = KFold(len(y), n_folds=10, shuffle=True)

    # cross validation here
    results_m = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # extract features for both sets of data
        for s in ['training', 'test']:
            if s == 'training':
                fs = X_train
            else:
                fs = X_test

            # extract features
            for feature in feature_names:
                # extract feature
                print("extracting " + feature + " feature for " + s + " set...", end='')

                # extract top m n-grams (and intervals)
                output = fe.extract_feature(fs, feature, s)

                # pickle the output
                pickle.dump(output, open("feature_pickle/" + s + "/" + feature + ".p", "wb"))

                print(" done.")


            # put together sets of features
            for feature_set in p_set[1:]:
                print("putting together the " + ", ".join(feature_set) + " feature set...", end='')
                feature_mat = []
                for feature in feature_set:
                    feature_vec = pickle.load(open("./feature_pickle/" + s + '/' + feature + ".p", "rb"))
                    feature_mat.extend(feature_vec)

                # transpose
                feature_mat_t = map(lambda *a: list(a), *feature_mat)

                # pickle the output
                pickle.dump(feature_mat_t, open("feature_set_pickle/" + s + "/" + "_".join(feature_set) + ".p", "wb"))

                print(" done.")


        models = ["svm", "logistic regression", "bayes", "lda"]

        # use the training set to train models
        for model in models:
            for feature_set in p_set[1:]:
                print("training " + model + " model using the " + ", ".join(feature_set) + " feature set...", end='')

                # load pickled feature set (could have easily not pickled it and just made it here...)
                feature_mat = pickle.load(open("feature_set_pickle/training/" + "_".join(feature_set) + ".p", "rb" ))

                # train model
                clf = train_model(model, feature_mat, y_train)

                # pickle the output
                pickle.dump(clf, open('model_pickle/' + model + '/' + "_".join(feature_set) + '.p', "wb"))

                print(" done.")


        # evaluate the models on the test set
        results = []
        for model in models:
            for feature_set in p_set[1:]:

                # load pickled feature model
                clf = pickle.load(open("model_pickle/" + model + '/' + "_".join(feature_set) + ".p", "rb" ))

                # load pickled feature set
                feature_mat = pickle.load(open("feature_set_pickle/test/" + "_".join(feature_set) + ".p", "rb" ))

                # get result
                y_t, y_p, c_m = predict_genre(clf, feature_mat, y_test)

                # keep track of results internally
                results.append((model, feature_set, (y_t, y_p, c_m)))

        results_m.append(results)

    # final output
    results_t = map(lambda *a: list(a), *results_m)
    out = []
    for t in results_t:
        y_ts = []
        y_ps = []
        conf_m = numpy.zeros((len(genres), len(genres)))
        for model, feature_set, (y_t, y_p, c_m) in t:
            y_ts.extend(y_t)
            y_ps.extend(y_p)
            conf_m = conf_m + c_m

        acc = accuracy_score(y_ts, y_ps)
        f1 = f1_score(y_ts, y_ps, average='weighted')
        out.append((t[0][0], t[0][1], (acc, f1, conf_m)))

    for exp in out:
        print(exp)

def train_model(model, X, y):
    if model == "svm":
        clf = svm.LinearSVC(dual=False, penalty='l1', class_weight="balanced", C=10) #C=10 determined experimentally
    elif model == "logistic regression":
        clf = linear_model.LogisticRegression(penalty='l1', class_weight="balanced", C=100) #C=100 determined experimentally
    elif model == "bayes":
        clf = naive_bayes.GaussianNB()
    elif model == "lda":
        clf = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    clf.fit(X, y)

    return clf


def predict_genre(clf, X_test, y_test):

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    #accuracy = accuracy_score(y_test, y_pred)
    #fone = f1_score(y_test, y_pred, average='weighted')

    return y_test, y_pred, cm


if __name__ == "__main__":
    main(sys.argv)
