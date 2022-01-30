from __future__ import division
import argparse
import pandas as pd
import random

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['Charles Boy De La Tour','Gabriel Drai','Zakariae El Asri']
__emails__  = ['charles.boy-de-la-tour@student-cs.fr','gabriel.drai@student-cs.fr','zakariae.elasri@student-cs.fr']

def text2sentences(path):
	# feel free to make a better tokenization/pre-processing
	sentences = []
	with open(path) as f:
		for l in f:
			l = ''.join(item.lower() for item in l if item.isalpha() or item == " ")
			l = l.split()
			l = [x for x in l if len(x) > 1]
			sentences.append(l)
	return sentences

def get_vocab(sentences):
	occurences = {}
	count = 0
	for sentence in sentences:
		for word in sentence:
			count += 1
			if word in occurences:
				occurences[word] += 1
			else:
				occurences[word] = 1
	sample = 0.001
	vocab = []
	for key in occurences:
		z = occurences[key]/count
		z = ((z/sample)**(1/2) + 1) * (sample/z)     
		if z >= 1:
			vocab.append(key)
	return vocab

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs

def word_to_index(vocab):
	w2id = {}
	id2w = {}
	count = 0
	for word in vocab:
		if word in w2id:
			pass
		else:
			w2id[word] = count
			id2w[count] = word
			count += 1
	return w2id, id2w


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
		
		self.trainset = sentences
		self.vocab = get_vocab(self.trainset)
		self.w2id, self.id2w = word_to_index(self.vocab)
		self.winSize = winSize
		self.nEmbed = nEmbed
		self.negativeRate = negativeRate
  
		self.l1 = np.random.randn(len(self.vocab), self.nEmbed)*0.01
		self.l2 = np.random.randn(self.nEmbed, len(self.vocab))*0.01

		self.all_losses = []


	def sample(self, omit):
		omit = (self.id2w[omit[0]], self.id2w[omit[1]])
		vocab_without_omit = [x for x in self.vocab if x not in omit]
		neg = random.sample(vocab_without_omit, self.negativeRate)
		neg = [self.w2id[x] for x in neg]
		return neg

	def train(self):
		self.trainWords = 0
		self.accLoss = 0.
		for sentence in self.trainset:
			sentence = list(filter(lambda word: word in self.vocab, sentence))
			for wpos, word in enumerate(sentence):
				wIdx = self.w2id[word]
				winsize = np.random.randint(self.winSize) + 1
				start = max(0, wpos - winsize)
				end = min(wpos + winsize + 1, len(sentence))

				ctxtIds = [self.w2id[x] for x in sentence[start:end] if x != word]

				for ctxtId in ctxtIds:
					negativeIds = 0#self.sample((wIdx, ctxtId))
					self.accLoss += self.trainWord(wIdx, ctxtId, negativeIds)
					self.trainWords += 1
		loss = self.accLoss/self.trainWords
		self.all_losses.append(loss)
		print("Cross Entropy Loss: ", loss)

	def trainWord(self, wordId, contextId, negativeIds):
		context = np.zeros((len(self.vocab),1))
		context[contextId] = 1
		#neg = np.zeros((len(self.vocab),1))
		#neg[negativeIds] = 1
		h = self.l1[wordId,:]
		h = h.reshape((len(h),1))
		y_pred = np.dot(self.l2.T, h)
		y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=0, keepdims=True)
		loss = y_pred - context# + neg
		dl2 = np.dot(loss, h.T)
		dl1 = np.dot(self.l2, loss)
		self.l1[wordId,:] -= 0.2 * dl1.T.reshape(dl1.shape[0])
		self.l2 -= 0.2 * dl2.T
		return -np.sum(np.log(y_pred[contextId]))
  
  
  
	def save(self, path):
		np.save(path, np.array([self.l1,
             					self.l2,
                  				self.all_losses,
                      			self.id2w,
                         		self.w2id],
                         		dtype=object),
          						allow_pickle=True)

	def similarity(self,word1,word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
		if word1 in self.w2id and word2 in self.w2id:
			word1 = self.w2id[word1]
			word2 = self.w2id[word2]
			h1 = self.l1[word1,:]
			h1 = h1.reshape((len(h1),1))
			h1 = np.dot(self.l2.T, h1)
			h1 = np.exp(h1) / np.sum(np.exp(h1), axis=0, keepdims=True)
			h2 = self.l1[word2,:]
			h2 = h2.reshape((len(h2),1))
			h2 = np.dot(self.l2.T, h2)
			h2 = np.exp(h2) / np.sum(np.exp(h2), axis=0, keepdims=True)
			return np.dot(h1.T, h2)[0][0]
		else:
			return 0

	def load(self, path):
		tmp = np.load(path+".npy", allow_pickle=True)
		self.l1 = tmp[0]
		self.l2 = tmp[1]
		self.all_losses = tmp[2]
		self.id2w = tmp[3]
		self.w2id = tmp[4]




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences)
		sg.train()
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram("")
		sg.load(opts.model)
		for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
			print('{:f}'.format(sg.similarity(a,b)))
