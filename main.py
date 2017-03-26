from __future__ import print_function
import sys
import midi
import os
import glob
import cPickle as pickle
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn import svm
import itertools
import operator
import numpy

import preprocess as pre
import feature_extraction as fe

def main(argv):

    genres = ['classical', 'past_century']

    # preprocess midi's if there is nothing in the preprocess_pickle folders
    for genre in genres:
        # build list of filenames
        filenames = []
        for ext in ['.mid', '.kar']:
            for f in os.listdir('data/' + genre):
                if f.endswith(ext):
                    filenames.append(os.path.join('data/' + genre + '/', f))

        print("preprocessing " + genre + " music files...")
        for filename in filenames:
            base = os.path.basename(filename)
            name = os.path.splitext(base)
            if not os.path.exists('preprocess_pickle/' + genre + '/' + name[0] + '.p'):
                print("\tpreprocessing " + name[0] + "...", end='')
                # read the midi file
                pattern = midi.read_midifile(filename)

                # preprocess the midi file
                out = pre.preprocess(pattern)

                # pickle the output
                pickle.dump(out, open("preprocess_pickle/" + genre + "/" + name[0] + ".p", "wb"))
                print(" done.")
            else:
                print("\tskipping " + name[0] + ".")
                continue

    feature_names = ['unigram', 'bigram', 'trigram', 'interval']

    # extract feature if its .p file is not in the feature_pickle folder
    for feature in feature_names:
        if not os.path.exists('feature_pickle/' + feature + '.p'):
            print("extracting " + feature + " feature...", end='')

            # extract top m n-grams
            output = fe.extract_feature(feature)

            # pickle the output
            pickle.dump(output, open("feature_pickle/" + feature + ".p", "wb"))

            print(" done.")
        else:
            print(feature + " feature is already extracted. skipping.")
            continue

    # make label vector
    labels = []
    for genre in genres:
        n = len(glob.glob(os.path.join('preprocess_pickle/' + genre + '/', '*.p')))
        labels.extend([genre] * n)

    # put together sets of features and give accuracy for each set

    # powerset of features
    p_set = list(itertools.chain.from_iterable(itertools.combinations(feature_names, r) for r in range(len(feature_names)+1)))

    # p_set = [[], feature_names]

    results = []
    for feature_set in p_set[1:]:
        if not os.path.exists('feature_set_pickle/' + "_".join(feature_set) + '.p'):
            print("scoring " + ", ".join(feature_set) + " feature set...", end='')
            feature_mat = []
            for feature in feature_set:
                feature_vec = pickle.load(open("./feature_pickle/" + feature + ".p", "rb"))
                feature_mat.extend(feature_vec)

            # transpose
            feature_mat_t = map(lambda *a: list(a), *feature_mat)

            # get result
            fone, accuracy, c_m = predict_genre(feature_mat_t, labels)

            # keep track of results internally
            results.append((feature_set, (fone, accuracy, c_m)))

            # pickle the output
            # probably better to do this as a .txt file or something
            pickle.dump((fone, accuracy, c_m), open("feature_set_pickle/" + "_".join(feature_set) + ".p", "wb"))

            print(" done.\nf1: " + str(round(fone, 4)) + "\nacc: " + str(round(accuracy, 4)) + "\ncm:\n" + str(c_m))
        else:
            (fone, accuracy, conf_m) = pickle.load(open("feature_set_pickle/" + "_".join(feature_set) + ".p", "rb"))
            results.append((feature_set, (fone, accuracy, conf_m)))
            print(", ".join(feature_set) + " feature set is already scored. result:\nf1: " + str(round(fone, 4)) + "\nacc: " + str(round(accuracy, 4)) + "\ncm:\n" + str(conf_m))
            continue

    # final output
    max_result = max(results, key=operator.itemgetter(1))
    print(", ".join(max_result[0]) + " scored the highest, result: " + str(round(max_result[1][0], 4)))

def predict_genre(X, y):
    X = numpy.array(X)
    y = numpy.array(y)

    kf = KFold(len(y), n_folds=10, shuffle=True)
    cm_sum = numpy.empty([2,2])

    accuracies = []
    fones = []
    for train_index, test_index in kf:
        clf = svm.SVC(kernel='linear', class_weight="balanced", C=3.0)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_sum += cm

        accuracies.append(accuracy_score(y_test, y_pred))
        fones.append(f1_score(y_test, y_pred, average='weighted'))

    return numpy.mean(fones), numpy.mean(accuracies), cm_sum

if __name__ == "__main__":
    main(sys.argv)
