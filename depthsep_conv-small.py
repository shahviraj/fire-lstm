
# coding: utf-8

# In[6]:


get_ipython().magic(u'matplotlib inline')


# In[7]:


from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np
import h5py
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape


# In[8]:


import cv2 as cv2


# In[27]:


comb_data = h5py.File('/local/viraj/courses/ME592x/assn3/Aditya_data/combustion_img_13.mat','r')
X = comb_data['train_set_x'][()]
y = comb_data['train_set_y'][()]


# In[28]:


X_trf, X_tsf,y_trf, y_tsf = train_test_split(X.T, y, test_size=0.4)


# In[50]:


#Resizing the image to lower resolution in (width, height): CHANGE THIS PARAMETER FOR DIFFERENT RESOLUTIONS
width = 100
height = 100
num_features = width*height


# In[51]:


a=np.shape(X_trf)[0]
X_train_final = np.zeros([a,num_features])
for i in range(0,a):
    temp_image = X_trf[i,:] 
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_image = cv2.resize(temp_image, dsize=((width,height)))
    temp_image = np.reshape(temp_image,[num_features,])
    X_train_final[i,:] = temp_image
    
b = np.shape(X_tsf)[0]
X_test_final = np.zeros([b,num_features])
for i in range(0,b):
    temp_image = X_tsf[i,:] 
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_image = cv2.resize(temp_image, dsize=(width,height))
    temp_image = np.reshape(temp_image,[num_features,])
    X_test_final[i,:] = temp_image


# In[52]:


X_train_final = X_train_final.reshape([a,height,width,1])
X_test_final = X_test_final.reshape([b,height,width,1])


# In[53]:


from keras.utils import to_categorical
y_trf_oh = to_categorical(y_trf)
y_tsf_oh = to_categorical(y_tsf)


# In[54]:


def SmallXception(input_shape=(100, 100, 1),
                  classes=2):
    img_input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = SeparableConv2D(128, (3, 3), use_bias=False)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), use_bias=False)(x)
    #x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = SeparableConv2D(256, (3, 3), use_bias=False)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), use_bias=False)(x)
    #x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    #x = SeparableConv2D(512, (3, 3), use_bias=False)(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = SeparableConv2D(512, (3, 3), use_bias=False)(x)
    #x = BatchNormalization()(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(img_input, x)
    return model


# In[55]:


if __name__ == '__main__':
    model = SmallXception()
    model.summary()
    
    print('Input image shape:', X_train_final.shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_final, y_trf_oh[0:a], epochs=1, batch_size=32)
   
    
    scores = model.evaluate(X_test_final, y_tsf_oh[0:b], verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

