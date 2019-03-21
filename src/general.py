import numpy as np
import re
import sklearn
import tensorflow as tf

def prepareData(train, dev, embeddings):	
	'''
	Almacenamiento de palabras en nuestro vocabulario -> Añadimos al vocabulario
	aquellas palabras que estén en el subconjunto de los embeddings seleccionados 
	'''

	#Expresión regular para usuario
	RE_TOKEN_USER = re.compile(r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")

	vocabulary = {}
	vocabulary["PADDING"] = len(vocabulary)
	vocabulary["UNKOWN"] = len(vocabulary)

	#Matriz de embeddings del vocabulario
	embeddings_matrix = []
	embeddings_matrix.append(np.zeros(100))
	embeddings_matrix.append(np.random.uniform(-0.25, 0.25, 100))

	for word in embeddings.wv.vocab:
		vocabulary[word] = len(vocabulary)
		#Al mismo tiempo creamos matrix de embeddings
		embeddings_matrix.append(embeddings[word])


	train_idx = []
	dev_idx = []

	for sentence in train:
		wordIndices = []
		for word in sentence:
			if RE_TOKEN_USER.fullmatch(word):
				word = "@user"

			if word in vocabulary:
				wordIndices.append(vocabulary[word])
			else:
				wordIndices.append(vocabulary["UNKOWN"])

		train_idx.append(np.array(wordIndices))

	for sentence in dev:
		wordIndices = []
		for word in sentence:
			if RE_TOKEN_USER.fullmatch(word):
				word = "@user"

			if word in vocabulary:
				wordIndices.append(vocabulary[word])
			else:
				wordIndices.append(vocabulary["UNKOWN"])

		dev_idx.append(np.array(wordIndices))

	return (train_idx, dev_idx, embeddings_matrix, vocabulary)


def prepareDataTest(test, vocabulary):
	'''
		Preparación de conjunto de test.
	'''

	RE_TOKEN_USER = re.compile(r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")

	test_idx = []

	for sentence in test:
		wordIndices = []
		for word in sentence:
			if RE_TOKEN_USER.fullmatch(word):
				word = "@user"

			if word in vocabulary:
				wordIndices.append(vocabulary[word])
			else:
				wordIndices.append(vocabulary["UNKOWN"])

		test_idx.append(np.array(wordIndices))

	return (test_idx)

def maxLength(data):
	'''
		Función que calcula la long máxima de las oraciones de entrada.
	'''

	max = 0

	for sentence in data:
		if len(sentence) > max:
			max = len(sentence)

	return max


def padding_truncate(sentences, max_length):
	'''
		Amplía o recorta las oraciones en función de max_length
	'''

	for i in range(len(sentences)):
		sent_size = len(sentences[i])

		if sent_size > max_length:
			sentences[i] = sentences[i][:max_length]
		elif sent_size < max_length:
			if(isinstance(sentences[i],list)):
				sentences[i] += [0] * (max_length - sent_size)
			else:
				list_sentence = sentences[i].tolist()
				list_sentence += [0] * (max_length - sent_size)
				sentences[i] = np.array(list_sentence)

	return sentences

def polarityToString(pol):
	if pol == 0:
		return "P"
	elif pol == 1:
		return "NEU"
	elif pol == 2:
		return "N"
	elif pol == 3:
		return "NONE"



def writeOutput(y_pred, id_, fichero):
	'''
		Generación de fichero de salida
	'''
	
	y_pred_tag = [polarityToString(pred) for pred in np.argmax(y_pred,axis=1)]
	output_data = zip(id_, y_pred_tag)

	with(open(fichero, 'w')) as f_test_out:
		f_test_out.write("Id,Expected")
		s_buff = "\n".join(["\t".join(list(label_pair)) for label_pair in output_data])
		f_test_out.write(s_buff)