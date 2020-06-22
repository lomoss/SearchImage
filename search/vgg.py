# -*- coding: utf-8 -*-
import keras
import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class VGGNet:
    def __init__(self):
        keras.backend.clear_session()
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        #加载自己训练的模型
        # self.model.load_weights(r'F:\model\vgg16_use6.h5',by_name=True)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        #归一化
        norm_feat = feat[0]/LA.norm(feat[0])
        # print(feat.shape)
        return norm_feat

if __name__ == '__main__':
    # url ='F:/图片/5e731215d78d5.jpg'
    vgg=VGGNet()
    # result =vgg.extract_feat(url)
    # print(result)