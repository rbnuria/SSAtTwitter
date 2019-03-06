#-*- coding: UTF-8 -*-

import xml.etree.ElementTree as ET
from nltk import word_tokenize, pos_tag, ne_chunk
import numpy as np


class XMLData:

	def __init__(self):
		self.data = []
		self.polarity = []
	
	def __init__(self, source, train):
		tree = ET.parse(source)
		self.data = []
		self.polarity = []

		for sentence in tree.iter('tweet'):
			text = sentence.find('content').text

			self.data.append(text)

			if train:
				polarity =  sentence.find('sentiment').find('polarity').find('value').text
				self.polarity.append(polarity)

	def getData(self):
		return np.array(self.data)

	def getPolarity(self):
		return np.array(self.polarity)




if __name__ == "__main__":
	source = '../data/intertass-ES-train-tagged.xml'
	object_ = XMLData(source).getData()