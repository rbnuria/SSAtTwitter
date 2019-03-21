import numpy as np

#Fijamos semilla
np.random.seed(666)
from tensorflow import set_random_seed
set_random_seed(2)

from XMLData import *
from general import prepareData, writeOutput, prepareDataTest, padding_truncate, maxLength
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, Flatten
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Embedding, BatchNormalization
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal, glorot_uniform
from gensim.models.keyedvectors import KeyedVectors
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers
import sklearn
import tensorflow as tf
import keras.backend as K

#Lectura de los datos
print("Leyendo datos de entrenamiento...")
xml_train = XMLData('../data/intertass-ES-train-tagged.xml',1)
data_train = xml_train.getData()
label_train = xml_train.getPolarity()

data_train_antiguo = data_train

print("Leyendo datos de desarrollo...")
xml_eval = XMLData('../data/intertass-ES-development-tagged.xml',1)
data_eval = xml_eval.getData()
label_eval = xml_eval.getPolarity()


print("Leyendo datos de test...")
xml_test = XMLData('../data/intertass-ES-test.xml',0)
data_test = xml_test.getData()
test_ids = xml_test.getIds()

print("Leyendo word embeddings...")
embeddings = KeyedVectors.load_word2vec_format('../data/fasttext_spanish_twitter_100d.vec', binary=False)


print("Transformamos las frases con los embeddings...")
data_train, data_eval, matrix_embeddings, vocab = prepareData(data_train, data_eval, embeddings)

data_test = prepareDataTest(data_test, vocab)

#PADDING-TRUNCATE
input_size = maxLength(data_train)

data_train = np.array(padding_truncate(data_train, input_size))
data_eval = np.array(padding_truncate(data_eval, input_size))
data_test = np.array(padding_truncate(data_test, input_size))

matrix_embeddings = np.array(matrix_embeddings)

print("Chequeo de dimensiones...")
print(data_train.shape)
print(label_train.shape)
print(data_eval.shape)
print(label_eval.shape)
print(data_test.shape)
print(matrix_embeddings.shape)

print(data_train_antiguo[1])
print(data_train[1])

 ###################### MODELO

sequence_input = Input(shape = (input_size, ), dtype = 'float64')
embedding_layer = Embedding(matrix_embeddings.shape[0], matrix_embeddings.shape[1], weights=[matrix_embeddings],trainable=False, input_length = input_size) #Trainable false
embedded_sequence = embedding_layer(sequence_input)

x = Dense(1)(embedded_sequence)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(4)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
preds = Activation('softmax')(x)

model = Model(sequence_input, preds)
model.summary()

#Las clases NONE y NEU son minoritarias. Si utilizáramos accuracy, se despreciarían --> Usamos F1_score
def f1_score(y_true, y_pred):
	y_pred = K.round(y_pred)
	tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
	fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
	fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

	p = tp / (tp + fp + K.epsilon())
	r = tp / (tp + fn + K.epsilon())

	f1 = 2*p*r / (p+r+K.epsilon())
	f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
	return K.mean(f1)

model.compile(loss='categorical_crossentropy', optimizer = 'adagrad', metrics=['accuracy', f1_score])

#earlyStopping = EarlyStopping('val_f1_score', patience=3, mode='min')

#modelo = model.fit(x = data_train, y = to_categorical(label_train,4), batch_size = 16, epochs = 30, validation_data=(data_eval, to_categorical(label_eval,4)), shuffle = False, callbacks=[earlyStopping])
modelo = model.fit(x = data_train, y = to_categorical(label_train,4), batch_size = 16, epochs = 30, validation_data=(data_eval, to_categorical(label_eval,4)), shuffle = False)
#loss, acc = model.evaluate(x=data_eval, y=to_categorical(label_eval,4), batch_size=16)

#Predecimos train
y_pred = model.predict(data_train, batch_size=16).argmax(axis=-1)
print("Accuracy train:")
print(sum(1 for x,y in zip(y_pred,label_train) if x == y) / len(y_pred))
print("F1 train:")
print(sklearn.metrics.f1_score(label_train, y_pred, average = "macro"))

#Predecimos evaluación
y_pred_eval = model.predict(data_eval, batch_size=16).argmax(axis=-1)
print("Accuracy eval:")
print(sum(1 for x,y in zip(y_pred_eval,label_eval) if x == y) / len(y_pred_eval))
print("F1 eval:")
print(sklearn.metrics.f1_score(label_eval, y_pred_eval, average = "macro"))

#Predecimos test y guardamos en archivo
y_pred_test = model.predict(data_test, batch_size=16)
writeOutput(y_pred_test, test_ids, "submision.csv")


