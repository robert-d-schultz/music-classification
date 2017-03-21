from __future__ import print_function
import sys
import midi
import os
import glob
import cPickle as pickle
from sklearn.model_selection import cross_val_score
from sklearn import svm
import itertools
import operator

import preprocess as pre
import feature_extraction as fe

def main(argv):

    genres = ['classical', 'past_decade']

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
            result = predict_genre(feature_mat_t, labels)

            # keep track of results internally
            results.append((feature_set, result))

            # pickle the output
            # probably better to do this as a .txt file or something
            pickle.dump(result, open("feature_set_pickle/" + "_".join(feature_set) + ".p", "wb"))

            print(" done. result: " + str(round(result, 2)))
        else:
            result = pickle.load(open("feature_set_pickle/" + "_".join(feature_set) + ".p", "rb"))
            results.append((feature_set, result))
            print(", ".join(feature_set) + " feature set is already scored. result: " + str(round(result, 2)))
            continue

    # final output
    max_result = max(results, key=operator.itemgetter(1))
    print(", ".join(max_result[0]) + " scored the highest, result: " + str(round(max_result[1],2)))

def predict_genre(X, Y):
	scores = cross_val_score(svm.SVC(kernel='linear', C=1.0), X, Y, scoring='accuracy', cv=10)
	return scores.mean()

if __name__ == "__main__":
    main(sys.argv)
