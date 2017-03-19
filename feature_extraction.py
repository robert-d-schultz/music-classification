import pickle
import nltk
import operator
import collections

def bag_of_ngrams(n, texts, l):
	ngramss = [list(nltk.ngrams(text, n)) for text in texts]
	ngram_counts = []
	ngram_sum = collections.defaultdict(lambda: 0)

	for ngrams in ngramss:
		ngram_counts.append(collections.defaultdict(lambda: 0))
		for ngram in ngrams:
			ngram_counts[-1][ngram] += 1
			ngram_sum[ngram] += 1

	ngram_sort = sorted(ngram_sum.items(), key=operator.itemgetter(1), reverse=True)

	out = []
	for ngram, _ in ngram_sort[:l]:
		counts = [ngram_count[ngram] for ngram_count in ngram_counts]
		out.append([float(counts[i]) / count_text[i] for i in range(0, len(texts))])
	return out

# extract features
# for ngrams it needs to:
# * import a pre_process pickle file
# * ngram it
# * add the ngrams to the master ngram count
# * once all files have been added, find top m ngrams
# * import the pre_process pickle files again
# * count the top ngrams in each
# * add those counts to final output

def extract_feature(feature):
	# unigrams
	if feature == "unigram"

	# bigrams
	elif feature == "bigram"

	# trigrams
	elif feature == "trigram"

	return out
