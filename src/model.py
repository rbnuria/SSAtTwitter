import numpy as np

#Fijamos semilla
np.random.seed(666)
from tensorflow import set_random_seed
set_random_seed(2)

from XMLData import *
from general import prepareData, writeOutput, prepareDataTest, padding_truncate
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal, glorot_uniform
from gensim.models.keyedvectors import KeyedVectors
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers


#Lectura de los datos
input_size = 20
print("Leyendo datos de entrenamiento...")
xml_train = XMLData('../data/intertass-ES-train-tagged.xml',1)
data_train = xml_train.getData()
label_train = xml_train.getPolarity()


print(data_train.shape)
print(label_train.shape)

print("Leyendo datos de desarrollo...")
xml_eval = XMLData('../data/intertass-ES-development-tagged.xml',1)
data_eval = xml_eval.getData()
label_eval = xml_eval.getPolarity()


print(data_eval.shape)
print(label_eval.shape)

print("Leyendo datos de test...")
data_test = XMLData('../data/intertass-ES-test.xml',0).getData()

print(data_test.shape)

print("Leyendo word embeddings...")
embeddings = KeyedVectors.load_word2vec_format('../data/fasttext-sbwc.vec.gz', binary=False)


print("Transformamos las frases con los embeddings...")
data_train, data_eval, matrix_embeddings, vocab = prepareData(data_train, data_eval, embeddings)

data_test = prepareDataTest(data_test, vocab)

#PADDING-TRUNCATE
data_train = np.array(padding_truncate(data_train, 20))
#label_train = np.array(padding_truncate(label_train, 20))
data_eval = np.array(padding_truncate(data_eval, 20))
#label_eval = np.array(padding_truncate(label_eval, 20))
data_test = np.array(padding_truncate(data_test, 20))


matrix_embeddings = np.array(matrix_embeddings)

print(data_train.shape)
print(label_train.shape)
print(data_eval.shape)
print(label_eval.shape)
print(data_test.shape)
print(matrix_embeddings.shape)
