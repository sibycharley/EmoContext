import sys
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\python35.zip")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\DLLs")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg")
print(sys.path)

import csv
import numpy as np
import pandas as pd
import scipy
import pickle
import matplotlib.pyplot as plt
import gensim
import string
import itertools
import json

from random import random
from random import randint
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras import regularizers, optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from functools import partial, update_wrapper


class SemEval_Clean():

	#Define the initialization Code
	def __init__(self, turn1, turn2, turn3, turn1d, turn2d, turn3d):
		
		#Readng the input data file
		self.TVec1 = pd.read_csv(turn1, header=None).as_matrix()
		self.TVec2 = pd.read_csv(turn2, header=None).as_matrix()
		self.TVec3 = pd.read_csv(turn3, header=None).as_matrix()

		self.TVec1D = pd.read_csv(turn1d, header=None).as_matrix()
		self.TVec2D = pd.read_csv(turn2d, header=None).as_matrix()
		self.TVec3D = pd.read_csv(turn3d, header=None).as_matrix()

		#Punctutaion Table	
		self.table = str.maketrans('', '', string.punctuation)

		#Max length of input sentences
		self.MaxLength = 15

	#Finding the Vocabulary
	def CreateVoc(self):

		TVec = np.append(np.append(self.TVec1[:,1], self.TVec2[:,1]), self.TVec3[:,1])
		TVecD = np.append(np.append(self.TVec1D[:,1], self.TVec2D[:,1]), self.TVec3D[:,1])

		TWord = []
		for i in range(TVec.shape[0]):

			print (i)

			if isinstance(TVec[i], str):
				TTotal = TVec[i].split()
				TPunc = [w.translate(self.table) for w in TTotal if isinstance(w, str) if w.isalpha()]
				TWord = np.append(TWord, [w.lower() for w in TPunc])

		for i in range(TVecD.shape[0]):

			print (i)

			if isinstance(TVecD[i], str):
				TTotal = TVecD[i].split()
				TPunc = [w.translate(self.table) for w in TTotal if isinstance(w, str) if w.isalpha()]
				TWord = np.append(TWord, [w.lower() for w in TPunc])	

		#Saving the Vocabulary
		unique_words = list(set(TWord))
		print (len(unique_words))
		np.savetxt('Vocabulary_Dev_v3.txt', unique_words, delimiter=',', fmt='%s', encoding='utf-8')	

	#Creating One Hot Vectors
	def CreateOneHot(self):

		#Vocabulary
		self.Voc = 20000

		#Reading the Vocabulary file
		TWord = pd.read_csv('Vocabulary_Dev_v3.txt', header=None, encoding='utf-8', delimiter='\n').as_matrix()
		TWord = TWord.ravel()

		TOne = [one_hot(d, self.Voc) for d in TWord]
		TOne = list(map(int, [list(itertools.chain(*TOne))][0]))
		W2I = dict(zip(TWord, TOne))
		json.dump(W2I, open('WIndex_Dev_v3.txt', 'w'))

		#Turn wise Encoding
		WVect1 = np.zeros((self.TVec1.shape[0], self.MaxLength))
		WVect2 = np.zeros((self.TVec2.shape[0], self.MaxLength))
		WVect3 = np.zeros((self.TVec3.shape[0], self.MaxLength))
		WVectF = np.zeros((self.TVec1.shape[0], self.MaxLength))		

		for i in range(self.TVec1.shape[0]):

			print (i)

			#Split into words
			if isinstance(self.TVec1[i,1], str):
				tokens_1 = self.TVec1[i,1].split()
			if isinstance(self.TVec2[i,1], str):
				tokens_2 = self.TVec2[i,1].split()
			if isinstance(self.TVec3[i,1], str):
				tokens_3 = self.TVec3[i,1].split()

			#Remove Punctuations
			words_1 = [w.translate(self.table) for w in tokens_1 if isinstance(w, str) if w.isalpha()]
			words_2 = [w.translate(self.table) for w in tokens_2 if isinstance(w, str) if w.isalpha()]
			words_3 = [w.translate(self.table) for w in tokens_3 if isinstance(w, str) if w.isalpha()]

			words_1 = [w.lower() for w in words_1]
			words_2 = [w.lower() for w in words_2]
			words_3 = [w.lower() for w in words_3]

			#Encoding
			encoding_docs_1 = [[W2I[d] for d in words_1 if (d != 'nan')]]
			encoding_docs_2 = [[W2I[d] for d in words_2 if (d != 'nan')]]
			encoding_docs_3 = [[W2I[d] for d in words_3 if (d != 'nan')]]	

			encoding_docs_F = [np.append(encoding_docs_1, encoding_docs_2)]
			encoding_docs_F = [np.append(encoding_docs_F, encoding_docs_3)]
			
			#Padding Sequences
			WVect1[i] = pad_sequences(encoding_docs_1, maxlen=self.MaxLength, padding='post')[0]
			WVect2[i] = pad_sequences(encoding_docs_2, maxlen=self.MaxLength, padding='post')[0]
			WVect3[i] = pad_sequences(encoding_docs_3, maxlen=self.MaxLength, padding='post')[0]
			WVectF[i] = pad_sequences(encoding_docs_F, maxlen=self.MaxLength, padding='post')[0]			


		#Turn wise Encoding for Dev
		WVect1D = np.zeros((self.TVec1D.shape[0], self.MaxLength))
		WVect2D = np.zeros((self.TVec2D.shape[0], self.MaxLength))
		WVect3D = np.zeros((self.TVec3D.shape[0], self.MaxLength))
		WVectFD = np.zeros((self.TVec1D.shape[0], self.MaxLength))		

		for i in range(self.TVec1D.shape[0]):

			print (i)

			#Split into words
			if isinstance(self.TVec1D[i,1], str):
				tokens_1 = self.TVec1D[i,1].split()
			if isinstance(self.TVec2D[i,1], str):
				tokens_2 = self.TVec2D[i,1].split()
			if isinstance(self.TVec3D[i,1], str):
				tokens_3 = self.TVec3D[i,1].split()

			#Remove Punctuations
			words_1 = [w.translate(self.table) for w in tokens_1 if isinstance(w, str) if w.isalpha()]
			words_2 = [w.translate(self.table) for w in tokens_2 if isinstance(w, str) if w.isalpha()]
			words_3 = [w.translate(self.table) for w in tokens_3 if isinstance(w, str) if w.isalpha()]

			words_1 = [w.lower() for w in words_1]
			words_2 = [w.lower() for w in words_2]
			words_3 = [w.lower() for w in words_3]

			#Encoding
			encoding_docs_1 = [[W2I[d] for d in words_1 if (d != 'nan')]]
			encoding_docs_2 = [[W2I[d] for d in words_2 if (d != 'nan')]]
			encoding_docs_3 = [[W2I[d] for d in words_3 if (d != 'nan')]]	

			encoding_docs_F = [np.append(encoding_docs_1, encoding_docs_2)]
			encoding_docs_F = [np.append(encoding_docs_F, encoding_docs_3)]			

			#Padding Sequences
			WVect1D[i] = pad_sequences(encoding_docs_1, maxlen=self.MaxLength, padding='post')[0]
			WVect2D[i] = pad_sequences(encoding_docs_2, maxlen=self.MaxLength, padding='post')[0]
			WVect3D[i] = pad_sequences(encoding_docs_3, maxlen=self.MaxLength, padding='post')[0]
			WVectFD[i] = pad_sequences(encoding_docs_F, maxlen=self.MaxLength, padding='post')[0]			


		print (self.TVec1[1000, 1])
		print (WVect1[1000])
		print (WVect1.shape)	
		print ("*********************************************************************")

		print (self.TVec2[1000, 1])
		print (WVect2[1000])
		print (WVect2.shape)
		print ("*********************************************************************")

		print (self.TVec3[1000, 1])
		print (WVect3[1000])
		print (WVect3.shape)	
		print ("*********************************************************************")				

		print (self.TVec1D[1000, 1])
		print (WVect1D[1000])
		print (WVect1D.shape)	
		print ("*********************************************************************")

		print (self.TVec2D[1000, 1])
		print (WVect2D[1000])
		print (WVect2D.shape)
		print ("*********************************************************************")

		print (self.TVec3D[1000, 1])
		print (WVect3D[1000])
		print (WVect3D.shape)	
		print ("*********************************************************************")						

		np.savetxt('Encode_T1_Full32.csv', WVect1, delimiter=',')
		np.savetxt('Encode_T2_Full32.csv', WVect2, delimiter=',')
		np.savetxt('Encode_T3_Full33.csv', WVect3, delimiter=',')
		np.savetxt('Encode_T1T2T3_Full33.csv', WVectF, delimiter=',')		

		np.savetxt('Encode_T1_Dev32.csv', WVect1D, delimiter=',')
		np.savetxt('Encode_T2_Dev32.csv', WVect2D, delimiter=',')
		np.savetxt('Encode_T3_Dev33.csv', WVect3D, delimiter=',')
		np.savetxt('Encode_T1T2T3_Dev33.csv', WVectFD, delimiter=',')		

		print ('Hello')

if __name__ == "__main__":

	#Define the file paths and directories
	Turn1 = "T1_v3.csv"
	Turn2 = "T2_v3.csv"
	Turn3 = "T3_v3.csv"
	Turn1D = "T1_Dev_v3.csv"
	Turn2D = "T2_Dev_v3.csv"
	Turn3D = "T3_Dev_v3.csv"
	Encode = 1

	#Call the Training constructor
	SEClean = SemEval_Clean(Turn1, Turn2, Turn3, Turn1D, Turn2D, Turn3D)

	if(Encode == 0):
		#Call the Vocabulary Creation
		SEClean.CreateVoc()
	else:
		#Call the OneHOt Creation
		SEClean.CreateOneHot()

	print ("Hello")	