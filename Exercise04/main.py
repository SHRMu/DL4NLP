from gensim.models.keyedvectors import KeyedVectors
from scipy.stats import spearmanr

def file_reader(file_path):
	word_list_1 = []
	word_list_2 = []
	score_list = []
	with open(file_path) as fp:
		for count, line in enumerate(fp):
			if count == 0:
				pass
			else:
				w1, w2, score = line.split("\t")
				score = score.split("\n")[0]
				word_list_1.append(w1)
				word_list_2.append(w2)
				score_list.append(float(score))
	return word_list_1, word_list_2, score_list
#file structure 
# |-- main.py
# |-- wordsim353
# |  	|--- combined.tab
# |-- GoogleNews-vectors-negative300.bin
if __name__ == '__main__':
	# problem 3.2a
	word_list_1, word_list_2, score_list = file_reader("./wordsim353/combined.tab")
	#print(word_list_1)
	#print(word_list_2)
	#print(score_list)

	# problem 3.2b
	result_score = []
	word2vec = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True, limit = 500000)
	for i in range(len(word_list_1)):
		#print(i)
		score = float(word2vec.similarity(word_list_1[i], word_list_2[i]))
		result_score.append(score)
	#print(result_score)

	# problem 3.2c
	coefficient = spearmanr(score_list, result_score)
	print("coefficient :", coefficient)

# SpearmanrResult(correlation=0.7000166486272194, pvalue=2.86866666051422e-53)
# the positive result means the positive correlationship between human results and word2vec results
# higher result means higher similarity between those human and word2vec results
# so 0.7 is not bad compared to the maximum positive correlation 1 which can be produced by Spearman’s rank correlation coefficient