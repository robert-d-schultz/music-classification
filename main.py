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

    genres = ['classical', 'past_century']
    feature_names = ['unigram', 'bigram', 'trigram', 'interval']
    # powerset of features (2^4 = 16)
    p_set = list(itertools.chain.from_iterable(itertools.combinations(feature_names, r) for r in range(len(feature_names)+1)))


    # separate data_unseparated into training and test sets
    if os.listdir('data/test/classical') == ['dummy']:
        for genre in genres:

            # build list of filenames
            filenames = []
            for ext in ['.mid', '.kar']:
                for f in os.listdir('data_unseparated/' + genre):
                    if f.endswith(ext):
                        filenames.append(os.path.join('data_unseparated/' + genre + '/', f))

            # shuffle filenames
            random.shuffle(filenames)

            # take 10% for test data
            test = filenames[int((len(filenames)+1)*.90):]

            # copy into /data/ folder
            for f in test:
                shutil.copy2(f, 'data/test/' + genre + '/')

            # take the rest for training data
            training = filenames[:int((len(filenames)+1)*.90)]

            # copy into /data/ folder
            for f in training:
                shutil.copy2(f, 'data/training/' + genre + '/')


    # preprocess and extract features for both sets of data
    for s in ['training', 'test']:

        # preprocess midi's (does both test set and training)
        for genre in genres:
            # build list of filenames
            filenames = []
            for ext in ['.mid', '.kar']:
                for f in os.listdir('data/' + s + '/' + genre):
                    if f.endswith(ext):
                        filenames.append(os.path.join('data/' + s + '/' + genre + '/', f))

            print("preprocessing " + genre + " music files for " + s + " set...")
            for filename in filenames:
                base = os.path.basename(filename)
                name = os.path.splitext(base)
                # preprocess midi if it doesn't have a corresponding .p file
                if not os.path.exists('preprocess_pickle/' + s + '/' + genre + '/' + name[0] + '.p'):
                    print("\tpreprocessing " + name[0] + "...", end='')
                    # read the midi file
                    pattern = midi.read_midifile(filename)

                    # preprocess the midi file
                    out = pre.preprocess(pattern)

                    # pickle the output
                    pickle.dump(out, open("preprocess_pickle/" + s + '/' + genre + "/" + name[0] + ".p", "wb"))
                    print(" done.")
                # otherwise skip
                else:
                    print("\tskipping " + name[0] + ".")
                    continue

        # extract features
        for feature in feature_names:
            # extract feature if its .p file is not in the feature_pickle folder
            if not os.path.exists('feature_pickle/' + s + '/' + feature + '.p'):
                print("extracting " + feature + " feature for " + s + " set...", end='')

                # extract top m n-grams (and intervals)
                output = fe.extract_feature(feature, s)

                # pickle the output
                pickle.dump(output, open("feature_pickle/" + s + "/" + feature + ".p", "wb"))

                print(" done.")
            else:
                print(feature + " feature is already extracted. skipping.")
                continue


        # put together sets of features
        for feature_set in p_set[1:]:
            if not os.path.exists('feature_set_pickle/' + s + "/" + "_".join(feature_set) + '.p'):
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
            else:
                print(",".join(feature_set) + " feature set is already put together. skipping.")
                continue


    models = ["svm", "logistic regression", "bayes", "lda"]

    # use the training set to train models
    for model in models:
        for feature_set in p_set[1:]:
            if not os.path.exists('model_pickle/' + model + '/' + "_".join(feature_set) + '.p'):
                print("training " + model + " model using the " + ", ".join(feature_set) + " feature set...", end='')

                # load pickled feature set (could have easily not pickled it and just made it here...)
                feature_mat = pickle.load(open("feature_set_pickle/training/" + "_".join(feature_set) + ".p", "rb" ))

                # make label vector
                labels = []
                for genre in genres:
                    n = len(glob.glob(os.path.join('preprocess_pickle/training/' + genre + '/', '*.p')))
                    labels.extend([genre] * n)

                # train model
                clf = train_model(model, feature_mat, labels)

                # pickle the output
                pickle.dump(clf, open('model_pickle/' + model + '/' + "_".join(feature_set) + '.p', "wb"))

                print(" done.")
            else:
                print(model + " was already trained with the " + ",".join(feature_set) + " feature set. skipping.")
                continue


    # evaluate the models on the test set
    results = []
    for model in models:
        for feature_set in p_set[1:]:

            # load pickled feature model
            clf = pickle.load(open("model_pickle/" + model + '/' + "_".join(feature_set) + ".p", "rb" ))

            # load pickled feature set
            feature_mat = pickle.load(open("feature_set_pickle/test/" + "_".join(feature_set) + ".p", "rb" ))

            # make label vector
            labels = []
            for genre in genres:
                n = len(glob.glob(os.path.join('preprocess_pickle/test/' + genre + '/', '*.p')))
                labels.extend([genre] * n)

            # get result
            fone, accuracy, c_m = predict_genre(clf, feature_mat, labels)

            # keep track of results internally
            results.append((model, feature_set, (fone, accuracy, c_m)))

    # final output
    # max_result = max(results, key=operator.itemgetter(2))
    # print(", ".join(max_result[0]) + " scored the highest using " + max_result[1] + ", f1: " + str(round(max_result[2][0], 4)))
    for result in results:
        print(result,end='\n')

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
    accuracy = accuracy_score(y_test, y_pred)
    fone = f1_score(y_test, y_pred, average='weighted')

    return fone, accuracy, cm

'''def predict_genre(model, X, y):
    X = numpy.array(X)
    y = numpy.array(y)

    kf = KFold(len(y), n_folds=10, shuffle=True)
    cm_sum = numpy.empty([2,2])

    accuracies = []
    fones = []
    for train_index, test_index in kf:
        if model == "svm":
            clf = svm.LinearSVC(dual=False, penalty='l1', class_weight="balanced", C=10) #C=10 determined experimentally
        elif model == "logistic regression":
            clf = linear_model.LogisticRegression(penalty='l1', class_weight="balanced", C=100) #C=100 determined experimentally
        elif model == "bayes":
            clf = naive_bayes.GaussianNB()
        elif model == "lda":
            clf = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_sum += cm

        accuracies.append(accuracy_score(y_test, y_pred))
        fones.append(f1_score(y_test, y_pred, average='weighted'))
    return numpy.mean(fones), numpy.mean(accuracies), cm_sum'''

if __name__ == "__main__":
    main(sys.argv)
