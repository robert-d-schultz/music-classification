# import music21 as mu

import sys
import midi
import os
import glob
import cPickle as pickle
import preprocess as pre
import feature_extraction as fe

def main(argv):
    # preprocess midi's if there is nothing in the preprocess_pickle folders
    for genre in ['classical', 'country']:
        if os.listdir('./preprocess_pickle/' + genre + '/') == []:
            # build list of filenames
            filenames = []
            for ext in ['.mid', '.kar']
                 filenames.extend(glob.glob(os.path.join('./data/' + genre + '/', '*' + ext)))

            for filename in filenames:

                # read the midi file
                pattern = midi.read_midifile(filename)

                # preprocess the midi file
                out = pre.preprocess(pattern)

                # print out
                #for i in out:
                #    print(i)

                # pickle the output
                base = os.path.basename(filename)
                name = os.path.splitext(base)
                pickle.dump(out, open("preprocess_pickle/" + genre + "/" + name[0] + ".p", "wb"))
        else:
            continue

    # extract feature if its .p file is not in the feature_pickle folder
    for feature in ['unigram', 'bigram', 'unigram']
        if not os.path.exists('./feature_pickle/' + feature + '.p'):
            # extract top m n-grams
            output = fe.extract_feature(feature)

            # pickle the output
            pickle.dump(output, open("feature_pickle/" + feature + ".p", "wb"))

if __name__ == "__main__":
    main(sys.argv)
