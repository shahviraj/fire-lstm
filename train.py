
# coding: utf-8

# In[4]:





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
from resizeimage import resizeimage
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.core import Flatten
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
comb_data = h5py.File('/local/viraj/courses/ME592x/assn3/Aditya_data/combustion_img_13.mat','r')
X = comb_data['train_set_x'][()]
y = comb_data['train_set_y'][()]


# In[13]:


np.shape(y.T)


# In[16]:


X_trf, X_tsf,y_trf, y_tsf = train_test_split(X.T, y, test_size=0.2)


# In[19]:


np.shape(X_trf)


# In[51]:


#Resizing the image to lower resolution: CHANGE THIS PARAMETER FOR DIFFERENT RESOLUTIONS
resize_resolution = (64,32)
num_features = np.prod(resize_resolution)


# In[53]:


X_train_final = np.zeros([43200,num_features])
a=43200
for i in range(0,a):
    temp_image = X_trf[i,:] 
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = cv2.resize(temp_image, dsize=resize_resolution)
    temp_image = np.reshape(temp_image,[num_features,])
    X_train_final[i,:] = temp_image
    
X_test_final = np.zeros([10800,num_features])
b=10800
for i in range(0,b):
    temp_image = X_tsf[i,:] 
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = cv2.resize(temp_image, dsize=resize_resolution)
    temp_image = np.reshape(temp_image,[num_features,])
    X_test_final[i,:] = temp_image
X_train_final = np.expand_dims(X_train_final, axis=2)
X_test_final = np.expand_dims(X_test_final, axis=2)


# In[46]:


np.shape(X_test_final)


# In[54]:



# fix random seed for reproducibility
np.random.seed(7)
model = Sequential()
#Convolution
model.add(Conv1D(input_shape= (num_features,1),filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#LSTM
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train_final, y_trf, epochs=1, batch_size=128)
# Final evaluation of the model
scores = model.evaluate(X_test_final, y_tsf, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[37]:





# In[ ]:




