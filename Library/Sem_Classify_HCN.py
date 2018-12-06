import sys
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\python35.zip")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\DLLs")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg")
print(sys.path)


import os
import numpy as np
import tensorflow as tf

#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(37)
#tf.set_random_seed(89)

import csv
import pandas as pd
import scipy
import pickle
import matplotlib.pyplot as plt
import gensim
import itertools
import json
import codecs

from tqdm import tqdm
from random import random
from random import randint
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from keras import regularizers, optimizers
from keras.utils import np_utils, to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Masking
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Dense, Lambda, LeakyReLU
from keras.layers import Dropout
from keras.models import load_model
from keras.layers import TimeDistributed, Concatenate, Average, GlobalMaxPooling1D, Activation
from keras.layers import Bidirectional, Flatten, GlobalAveragePooling1D, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras import regularizers, optimizers
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.constraints import unit_norm
from keras.layers.core import Reshape
from functools import partial, update_wrapper
from keras.engine.topology import Layer

class AttLayer2(Layer):
	def __init__(self, attention_dim, **kwargs):
		self.attention_dim = attention_dim
		super(AttLayer2, self).__init__()

	def build(self, input_shape):
		assert len(input_shape) == 3
		self.W = self.add_weight(name='Weight', shape=(input_shape[-1], self.attention_dim), initializer='normal', trainable=True)
		self.b = self.add_weight(name='Bias', shape=(self.attention_dim, ), initializer='normal', trainable=True)
		self.u = self.add_weight(name='Context', shape=(self.attention_dim, 1), initializer='normal', trainable=True)
		super(AttLayer2, self).build(input_shape)

	def call(self, x):
		# size of x :[batch_size, sel_len, attention_dim]
		# uit = tanh(xW+b)

		uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
		uit = K.dot(uit, self.u)
		uit = K.squeeze(uit, -1)
		alpha = K.exp(uit)
		alpha = alpha/K.cast(K.sum(alpha, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		alpha = K.expand_dims(alpha)
		
#		print (x.shape)
#		print (alpha.shape)

		weighted_state = x * alpha
		output = K.sum(weighted_state, axis=1)

		return output

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

	def get_config(self):
		config = {'attention_dim': self.attention_dim}
		base_config = super(AttLayer2, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class SemEval_Classify():

	#Define the initialization Code
	def __init__(self, WVect1, WVect2, WVect3, WVectF, label, WVect1D, WVect2D, WVect3D, WVectFD):
		
		#Readng the input data file
		self.WT1 = pd.read_csv(WVect1, header=None).as_matrix()
		self.WT2 = pd.read_csv(WVect2, header=None).as_matrix()
		self.WT3 = pd.read_csv(WVect3, header=None).as_matrix()
		self.WTF = pd.read_csv(WVectF, header=None).as_matrix()
		self.lab = pd.read_csv(label).as_matrix()

		self.WT1D = np.zeros((2816, 10))
		self.WT1D[0:2755,:] = pd.read_csv(WVect1D, header=None).as_matrix()

		self.WT2D = np.zeros((2816, 10))
		self.WT2D[0:2755,:] = pd.read_csv(WVect2D, header=None).as_matrix()
		
		self.WT3D = np.zeros((2816, 10))
		self.WT3D[0:2755,:] = pd.read_csv(WVect3D, header=None).as_matrix()

		self.WTFD = np.zeros((2816, 15))
		self.WTFD[0:2755,:] = pd.read_csv(WVectFD, header=None).as_matrix()

	#Classify Execution and Training
	def SEExecute(self):

		#Batch Size
		bSize = 128

		#Max Length of a Sentence
		actStep = 10

		#Total Samples
		vsamp = self.WT1.shape[0]

		#Converting the labels from categories to integers
		self.lab[self.lab == 'others'] = 0
		self.lab[self.lab == 'angry'] = 1
		self.lab[self.lab == 'happy'] = 2
		self.lab[self.lab == 'sad'] = 3

		YTemp = self.lab

		#Sampling to get correct ratio in classes
		Y0idx = np.where(YTemp == 0)[0]
		Y1idx = np.where(YTemp == 1)[0]
		Y2idx = np.where(YTemp == 2)[0]
		Y3idx = np.where(YTemp == 3)[0]

		#Collecting the testing data
		XTest_1 = self.WT1D
		XTest_2 = self.WT2D
		XTest_3 = self.WT3D
		XTest_F = self.WTFD		

		#Populating X and Y for Train/Test splitting
		X = list(range(vsamp))
		Y = self.lab
		Idx = X
		XTr, XVa, YTr, YVa, IdxTr, IdxVa = train_test_split(X, Y, Idx, stratify=Y, test_size=0.20)

		#Collecting the data based on split - Training
		XTrain_1 = self.WT1[IdxTr]
		XTrain_2 = self.WT2[IdxTr]
		XTrain_3 = self.WT3[IdxTr]
		XTrain_F = self.WTF[IdxTr]
		YTrain = Y[IdxTr]

		YTr = Y[IdxTr]
		YTr = np.transpose(list(itertools.chain(*YTr)))
		YTr = YTr[0:24064]

		XTrain_1 = XTrain_1[0:24064].astype(int)
		XTrain_2 = XTrain_2[0:24064].astype(int)
		XTrain_3 = XTrain_3[0:24064].astype(int)
		XTrain_F = XTrain_F[0:24064].astype(int)		
		YTrain = to_categorical(YTrain[0:24064], num_classes=4)

		#Collecting the data based on split - Validation
		XValid_1 = self.WT1[IdxVa]
		XValid_2 = self.WT2[IdxVa]
		XValid_3 = self.WT3[IdxVa]
		XValid_F = self.WTF[IdxVa]		
		YValid = Y[IdxVa]

		YVa = Y[IdxVa]
		YVa = np.transpose(list(itertools.chain(*YVa)))

		#Sub sampling the validation dataset to get correct ratio in classes		 
		Y0idx = np.where(YVa == 0)[0]
		Y1idx = np.where(YVa == 1)[0]
		Y2idx = np.where(YVa == 2)[0]
		Y3idx = np.where(YVa == 3)[0]

		Yidx = np.append(np.append(np.append(Y0idx[0:2358], Y1idx[0:110]), Y2idx[0:110]), Y3idx[0:110])
		YVa = YVa[Yidx]

		XValid_1 = XValid_1[Yidx].astype(int)
		XValid_2 = XValid_2[Yidx].astype(int)
		XValid_3 = XValid_3[Yidx].astype(int)
		XValid_F = XValid_F[Yidx].astype(int)		
		YValid = to_categorical(YValid[Yidx], num_classes=4)

		print (np.sum(YTr == 0) / len(YTr))
		print (np.sum(YTr == 1) / len(YTr))
		print (np.sum(YTr == 2) / len(YTr))
		print (np.sum(YTr == 3) / len(YTr))

		print (np.sum(YVa == 0) / len(YVa))
		print (np.sum(YVa == 1) / len(YVa))
		print (np.sum(YVa == 2) / len(YVa))
		print (np.sum(YVa == 3) / len(YVa))

		print (len(YTr))
		print (len(YVa))

		#Flipping the Train and Valid Data
#		XTrain_1 = np.fliplr(XTrain_1)
#		XTrain_2 = np.fliplr(XTrain_2)
#		XTrain_3 = np.fliplr(XTrain_3)

#		XValid_1 = np.fliplr(XValid_1)
#		XValid_2 = np.fliplr(XValid_2)
#		XValid_3 = np.fliplr(XValid_3)

		print (XTrain_3.shape)
		print (XValid_3.shape)

		count = 0

		##########################################################
		#Word Vectors based on Pre Trained GloVe
		##########################################################

		#Reading the 100 dimensional word vectors from GloVe
		embedding_matrix = pd.read_csv('./GloVe/glove_Embed_100d.csv', header=None).as_matrix()		

		##########################################################
		#Defining the Network
		##########################################################

		#Input Dimension(Vocabulary)
		iDim = 20000

		#Embedding Dimensions
		Edim = 100

		#Input Layer - Encoder
		Seqin_1 = Input(batch_shape=(bSize, actStep))
		Seqin_2 = Input(batch_shape=(bSize, actStep))
		Seqin_3 = Input(batch_shape=(bSize, actStep))		

		#Embedding Layer - Encoder
#		Embed_1 = Embedding(input_dim=iDim, output_dim=Edim, input_length=actStep, mask_zero=False, weights=[embedding_matrix], trainable=True, embeddings_constraint=unit_norm())(Seqin_1)
#		Embed_2 = Embedding(input_dim=iDim, output_dim=Edim, input_length=actStep, mask_zero=False, weights=[embedding_matrix], trainable=True, embeddings_constraint=unit_norm())(Seqin_2)		
#		Embed_3 = Embedding(input_dim=iDim, output_dim=Edim, input_length=actStep, mask_zero=False, weights=[embedding_matrix], trainable=True, embeddings_constraint=unit_norm())(Seqin_3)

		ELayer = Embedding(input_dim=iDim, output_dim=Edim, input_length=actStep, weights=[embedding_matrix], trainable=True)

		Xcon1 = []
		Xcon2 = []
		Xcon3 = []

		#Convolution Layer - 1
		fSize = [3,4]
		for fil in fSize:
			cTemp = Conv1D(nb_filter=32, filter_length=fil)(ELayer(Seqin_1))
			bTemp = BatchNormalization()(cTemp)
			aTemp = Activation('relu')(bTemp)
			pTemp = MaxPooling1D(2)(aTemp)
			Xcon1.append(pTemp)

			cTemp = Conv1D(nb_filter=32, filter_length=fil)(ELayer(Seqin_2))
			bTemp = BatchNormalization()(cTemp)
			aTemp = Activation('relu')(bTemp)
			pTemp = MaxPooling1D(2)(aTemp)
			Xcon2.append(pTemp)

			cTemp = Conv1D(nb_filter=32, filter_length=fil)(ELayer(Seqin_3))
			bTemp = BatchNormalization()(cTemp)
			aTemp = Activation('relu')(bTemp)
			pTemp = MaxPooling1D(2)(aTemp)
			Xcon3.append(pTemp)

		XFlat1 = Concatenate(axis=1)(Xcon1)	
		XFlat2 = Concatenate(axis=1)(Xcon2)	
		XFlat3 = Concatenate(axis=1)(Xcon3)	

		#Convolution Layer - 2
		cTemp = Conv1D(nb_filter=64, filter_length=2)(XFlat1)
		bTemp = BatchNormalization()(cTemp)
		aTemp = Activation('relu')(bTemp)
		Xtmp = MaxPooling1D(2)(aTemp)
		XFlat1 = Flatten()(Xtmp)

		cTemp = Conv1D(nb_filter=64, filter_length=2)(XFlat2)
		bTemp = BatchNormalization()(cTemp)
		aTemp = Activation('relu')(bTemp)
		Xtmp = MaxPooling1D(2)(aTemp)
		XFlat2 = Flatten()(Xtmp)		

		cTemp = Conv1D(nb_filter=64, filter_length=2)(XFlat3)
		bTemp = BatchNormalization()(cTemp)
		aTemp = Activation('relu')(bTemp)
		Xtmp = MaxPooling1D(2)(aTemp)				
		XFlat3 = Flatten()(Xtmp)			

		print (XFlat1.shape)

		#FCN Layer
		XUni1 = Dense(200, activation='relu')(XFlat1)
		XUni2 = Dense(200, activation='relu')(XFlat2)
		XUni3 = Dense(200, activation='relu')(XFlat3)						

		print (XUni1.shape)

		XUni1 = Reshape((1, int(XUni1.shape[1])))(XUni1)
		XUni2 = Reshape((1, int(XUni2.shape[1])))(XUni2)
		XUni3 = Reshape((1, int(XUni3.shape[1])))(XUni3)

		print (XUni1.shape)

		XUni = Concatenate(axis=1)([XUni1, XUni2, XUni3])

		print (XUni.shape)
#		exit()

		XTemp = Bidirectional(GRU(64, kernel_initializer='he_normal'))(XUni)
		XAtt = Dropout(0.3)(XTemp)

#		XAtt = AttLayer2(128)(XEnc)		

		#Fully Connected
		Xout = Dense(4, activation='softmax')(XAtt)

		model = Model(inputs=[Seqin_1, Seqin_2, Seqin_3], outputs=Xout)
		model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])

		print (model.summary())

		count = 0
		maxf1 = 1
		while(count < 10):
		
			#Fitting the model on the sequential data
			Mod = model.fit([XTrain_1, XTrain_2, XTrain_3], YTrain, validation_data=([XValid_1, XValid_2, XValid_3], YValid), epochs=1, batch_size=bSize, verbose=2)						

			count = count + 1		

			loss = Mod.history['val_loss'][0]
			print (loss)

			if(maxf1 > loss):
				maxf1 = loss

				print ('Saving Start')
				model.save('SemMod_HCN.h5')
				print ('Saving Stop')

			val2 = model.predict([XValid_1, XValid_2, XValid_3], batch_size=bSize, verbose=2)
			res = val2.argmax(axis=-1)
			print (roc_auc_score(YValid, val2))
			f1 = f1_score(YVa, res, labels=[1, 2, 3], average='micro')
			print (f1)

		model = load_model('SemMod_HCN.h5', custom_objects={'AttLayer2' :  AttLayer2})		

		#Evaluation and Prediction
		scores1 = model.evaluate([XTrain_1, XTrain_2, XTrain_3], YTrain, batch_size=bSize, verbose=2)
		val1 = model.predict([XTrain_1, XTrain_2, XTrain_3], batch_size=bSize, verbose=2)

		scores2 = model.evaluate([XValid_1, XValid_2, XValid_3], YValid, batch_size=bSize, verbose=2)
		val2 = model.predict([XValid_1, XValid_2, XValid_3], batch_size=bSize, verbose=2)


		print ("****************************************************************************")
		res = val1.argmax(axis=-1)

		print (scores1[1])
		print (roc_auc_score(YTrain, val1))
		print (f1_score(YTr, res, labels=[1, 2, 3], average='micro'))

		print ("****************************************************************************")
		res = val2.argmax(axis=-1)

		print (scores2[1])
		print (roc_auc_score(YValid, val2))
		print (f1_score(YVa, res, labels=[1, 2, 3], average='micro'))		

		print (np.sum(res == 1))
		print (np.sum(res == 2))
		print (np.sum(res == 3))
		print (np.sum(res == 0))

		print ("****************************************************************************")

		#Prediction - Development
		valD = model.predict([XTest_1, XTest_2, XTest_3], batch_size=bSize, verbose=2)
		resD = valD.argmax(axis=-1)		

		print (np.sum(resD == 1))
		print (np.sum(resD == 2))
		print (np.sum(resD == 3))
		print (np.sum(resD == 0))

		print ("****************************************************************************")		

if __name__ == "__main__":

	#Define the file paths and directories
	WVect1 = "./Encode/T1_Full.csv"
	WVect2 = "./Encode/T2_Full.csv"
	WVect3 = "./Encode/T3_Full.csv"
	WVectF = "./Encode/T1T2T3_Full.csv"
	label = "./Encode/Label.csv"

	WVect1D = "./Encode/T1_Dev.csv"
	WVect2D = "./Encode/T2_Dev.csv"
	WVect3D = "./Encode/T3_Dev.csv"
	WVectFD = "./Encode/T1T2T3_Dev.csv"	

	#Call the Training constructor
	SEClassify = SemEval_Classify(WVect1, WVect2, WVect3, WVectF, label, WVect1D, WVect2D, WVect3D, WVectFD)

	#Call the Model training Execution
	SEClassify.SEExecute()

	print ("Hello")	