# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import cv2
import pickle
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf
from keras.losses import categorical_crossentropy
from utils import *
from keras import Model
import numpy as np

def train_data_loader(data_path, img_size, output_path):
    label_list = []
    img_list = []
    label_idx = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            #try:
            #    img = cv2.imread(img_path, 1)
            #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #    img = cv2.resize(img, img_size)
            #except:
            #    continue
            label_list.append(label_idx)
            img_list.append(img_path)
        label_idx += 1

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_list, img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_list, label_f)

def pca_model_loader(data_path, model,img_size, output_path):
    print('start pca_model_loader')
    L=3
    batch_size=200
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('GAP_LAST').input)
    pca_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32', samplewise_center=True, samplewise_std_normalization=True)
    pca_generator = pca_datagen.flow_from_directory(directory=data_path, target_size=img_size,  color_mode="rgb",  batch_size=batch_size,  class_mode=None,  shuffle=True)
    featuresList = intermediate_layer_model.predict_generator(pca_generator, steps=len(pca_generator)//100,workers=4, verbose=1)
    print('extract features complete:',featuresList.shape)
    PCAMAC = extractRMAC(featuresList, intermediate_layer_model, True, L)
    print('extract RMAC:',len(PCAMAC))
    W, Xm = learningPCA(PCAMAC)

    # write output file for caching
    with open(output_path[0], 'wb') as pca_w:
        pickle.dump(W, pca_w)
    with open(output_path[1], 'wb') as pca_xm:
        pickle.dump(Xm, pca_xm)


# nsml test_data_loader
def test_data_loader(data_path):
    data_path = os.path.join(data_path, 'test', 'test_data')

    # return full path
    queries_path = [os.path.join(data_path, 'query', path) for path in os.listdir(os.path.join(data_path, 'query'))]
    references_path = [os.path.join(data_path, 'reference', path) for path in
                       os.listdir(os.path.join(data_path, 'reference'))]

    return queries_path, references_path


if __name__ == '__main__':
    query, refer = test_data_loader('./')
    print(query)
    print(refer)
