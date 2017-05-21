import cPickle as pickle
import glob
import os
import nltk
import operator
import collections
import numpy


# extraction only for training set
def bag_of_ngrams(filenames, feature, m):

	if feature == "unigram":
		n = 1
	if feature == "bigram":
		n = 2
	if feature == "trigram":
		n = 3

	song_chord_counts = []
	song_ngram_counts = []
	overall_ngram_count = collections.defaultdict(lambda: 0)

	for filename in filenames:

		# open pickled array
		data = pickle.load(open(filename, "rb" ))

		# chord only
		data = [a[1] for a in data]

		# n-grams
		data_ngrams = list(nltk.ngrams(data, n))

		# add a new dictionary for the new song
		song_ngram_counts.append(collections.defaultdict(lambda: 0))

		# go through each ngram, adding to the per-song count and the overall count
		for ngram in data_ngrams:
			song_ngram_counts[-1][ngram] += 1
			overall_ngram_count[ngram] += 1

		# keep track of number of chords
		song_chord_counts.append(len(data))

	# sort the overall count in descending order
	overall_ngram_sort = sorted(overall_ngram_count.items(), key=operator.itemgetter(1), reverse=True)

	out = []
	for ngram, _ in overall_ngram_sort[:m]:
		counts = [song_ngram_count[ngram] for song_ngram_count in song_ngram_counts]
		out.append([float(counts[i]) / song_chord_counts[i] for i in range(len(filenames))])

	# pickle which ngrams were used (need this for the test set)
	pickle.dump([ngram for ngram, _ in overall_ngram_sort[:m]], open("feature_pickle/" + feature + "_master.p", "wb"))

	return out

# extract for test set
def bag_of_ngrams_test(filenames, feature):
	# open the ngrams that need to be extracted
	n = 1
	ngram_master = pickle.load(open("feature_pickle/" + feature + "_master.p", "rb" ))
	if feature == "unigram":
		n = 1
	if feature == "bigram":
		n = 2
	if feature == "trigram":
		n = 3

	song_chord_counts = []
	song_ngram_counts = []

	for filename in filenames:

		# open pickled array
		data = pickle.load(open(filename, "rb" ))

		# chord only
		data = [a[1] for a in data]

		# n-grams
		data_ngrams = list(nltk.ngrams(data, n))

		# add a new dictionary for the new song
		song_ngram_counts.append(collections.defaultdict(lambda: 0))

		# go through each ngram, adding to the per-song count
		for ngram in data_ngrams:
			song_ngram_counts[-1][ngram] += 1

		# keep track of number of chords
		song_chord_counts.append(len(data))

	out = []
	for ngram in ngram_master:
		counts = [song_ngram_count[ngram] for song_ngram_count in song_ngram_counts]
		out.append([float(counts[i]) / song_chord_counts[i] for i in range(len(filenames))])

	return out

def avg_interval(filenames):

	avgs = []
	stdevs = []
	for filename in filenames:
		# open pickled array
		data = pickle.load(open(filename, "rb" ))

		# get all the intervals for a single song
		intervals = [data[0][0]]
		for i in range(1,len(data)):
			intervals.append(data[i][0] - data[i-1][0])

		avg = numpy.mean(intervals, dtype=numpy.float32)
		avgs.append(avg)
		stdev = numpy.std(intervals, dtype=numpy.float32)
		stdevs.append(stdev)

	return [avgs, stdevs]


def extract_feature(filenames, feature, s):
	out = []
	# unigrams
	if feature == "unigram":
		if s == "test":
			out = bag_of_ngrams_test(filenames, "unigram")
		else:
			out = bag_of_ngrams(filenames, "unigram", 120) # 112 named chords + 7 notes + 1 rest
	# bigrams
	elif feature == "bigram":
		if s == "test":
			out = bag_of_ngrams_test(filenames, "bigram")
		else:
			out = bag_of_ngrams(filenames, "bigram", 100) # fairly arbitrary
	# trigrams
	elif feature == "trigram":
		if s == "test":
			out = bag_of_ngrams_test(filenames, "trigram")
		else:
			out = bag_of_ngrams(filenames, "trigram", 500) # more than bigrams because 120x more
	# interval
	elif feature == "interval":
		out = avg_interval(filenames)

	return out
