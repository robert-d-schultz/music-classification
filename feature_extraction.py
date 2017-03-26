import cPickle as pickle
import glob
import os
import nltk
import operator
import collections
import numpy


def bag_of_ngrams(n, m):

	song_chord_counts = []
	song_ngram_counts = []
	overall_ngram_count = collections.defaultdict(lambda: 0)

	# build file list
	filenames = []
	for genre in ['classical', 'past_century']:
		filenames.extend(glob.glob(os.path.join('./preprocess_pickle/' + genre + '/', '*.p')))

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

	return out


def avg_interval():

	# build file list
	filenames = []
	for genre in ['classical', 'past_century']:
		filenames.extend(glob.glob(os.path.join('./preprocess_pickle/' + genre + '/', '*.p')))

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


def extract_feature(feature):
	out = []
	# unigrams
	if feature == "unigram":
		out = bag_of_ngrams(1, 119)
	# bigrams
	elif feature == "bigram":
		out = bag_of_ngrams(2, 100)
	# trigrams
	elif feature == "trigram":
		out = bag_of_ngrams(3, 500)
	# interval
	elif feature == "interval":
		out = avg_interval()

	return out
