#-*- coding: UTF-8 -*-

import xml.etree.ElementTree as ET
from nltk import word_tokenize, pos_tag, ne_chunk
import numpy as np
from nltk.tokenize.casual import TweetTokenizer

class XMLData:

	def __init__(self):
		self.data = []
		self.polarity = []
	
	def __init__(self, source, train):
		tree = ET.parse(source)
		self.data = []
		self.polarity = []
		self.id = []

		#Tokenizador tweets
		TWEET_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)


		for sentence in tree.iter('tweet'):
			id_ = sentence.find('tweetid').text
			text = TWEET_TOKENIZER.tokenize(sentence.find('content').text)

			self.data.append(text)
			self.id.append(id_)

			if train:
				polarity =  sentence.find('sentiment').find('polarity').find('value').text
				self.polarity.append(self.polarityToInt(polarity))

	def getData(self):
		return np.array(self.data)

	def getPolarity(self):
		return np.array(self.polarity)

	def getIds(self):
		return np.array(self.id)

	def polarityToInt(self, pol):
		if pol == "P":
			return 0
		elif pol == "NEU":
			return 1
		elif pol == "N":
			return 2
		elif pol == "NONE":
			return 3

