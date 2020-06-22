#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
import numpy as np  
seed = 8
np.random.seed(seed)
from keras.applications import VGG16
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
from tqdm import tqdm_notebook
from random import shuffle
import shutil 
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense,Activation,GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten

from keras.constraints import maxnorm
from keras.optimizers import SGD 
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
train_dir="C:/Users/安之/Desktop/keras_vgg16/data/Image_data/test"

val_dir="C:/Users/安之/Desktop/keras_vgg16/data/Image_data/train"

test_dir="C:/Users/安之/Desktop/keras_vgg16/data/Image_data/validation"


train_pic_gen=ImageDataGenerator(rescale=1./255)

test_pic_gen=ImageDataGenerator(rescale=1./255)

val_pic_gen=ImageDataGenerator(rescale=1./255)

train_flow=train_pic_gen.flow_from_directory(train_dir,(224,224),batch_size=8,class_mode='categorical')

val_flow=val_pic_gen.flow_from_directory(val_dir,(224,224),batch_size=8,class_mode='categorical')

test_flow=test_pic_gen.flow_from_directory(test_dir,(224,224),batch_size=8,class_mode='categorical')

print(train_flow.class_indices)

model = Sequential()  
model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(20,activation='softmax'))
#model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  

model.compile(optimizer=SGD(lr=0.00001, momentum=0.9,decay=0.00005),loss='categorical_crossentropy',metrics=['acc'])
#model.compile(optimizer=SGD(lr=1e-6, momentum=0.9,decay=1e-6),loss='categorical_crossentropy',metrics=['acc'])
#import catvsdogs.morph as mp

history=model.fit_generator(
      train_flow,
      steps_per_epoch=30,#每轮训练次
      epochs=50,#轮次
      validation_data=val_flow,
      validation_steps=12,callbacks=[TensorBoard(log_dir='D:/model_pro/log3')])
model.save('D:/model_pro/model/vgg16_use7.h5')
test_loss, test_acc = model.evaluate_generator(test_flow, 96)

print('test acc:', test_acc)