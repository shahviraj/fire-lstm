
# coding: utf-8

# In[ ]:



# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import scipy as sc
import cv2 as cv2
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow
from PIL import Image
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
comb_data = h5py.File('../fire-dataset/combustion_img_13.mat','r')
X = comb_data['train_set_x'][()]
y = comb_data['train_set_y'][()]
X_trf, y_trf, X_tsf, y_tsf = train_test_split(X, y, test_size=0.2)
X_train_final = np.zeros([43200,])
a=len(X_train_final[:,1])
for i in range(0,a):
    temp_image = X_trf[:,i] 
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = cv2.resizeimage((64, 32), temp_image)
    temp_image = np.reshape(temp_image,[1204,1])
    X_train_final[i,:] = temp_image
X_test_final = np.zeros([10800,])
b=len(X_test_final[:,1])
for i in range(0,b):
    temp_image = X_tsf[:,i] 
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = cv2.resizeimage((64, 32), temp_image)
    temp_image = np.reshape(temp_image,[1204,1])
    X_test_final[i,:] = temp_image    
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train_final = sequence.pad_sequences(X_train_final, maxlen=max_review_length)
X_test_final = sequence.pad_sequences(X_test_final, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test_final, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

