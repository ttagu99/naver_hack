# -*- coding: utf_8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

import cv2
import argparse
import pickle

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation,Average
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import backend as K
from data_loader import train_data_loader
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model,load_model
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa
import random
from keras.utils.training_utils import multi_gpu_model

def get_tta_image(image, tta):
    images=[]
    temp = image
    images.append(temp)
    if tta == 8:
        for i in range(3):
            temp = np.rot90(temp)
            images.append(temp)
        for i in range(len(images)):
            images.append(np.fliplr(images[i])) 
    return images

def predict_tta(model, img, tta=8, use_avg=False):
    img_ttas = get_tta_image(img,tta)
    outputs = []
    for img_tta in img_ttas:
        img_tta = np.expand_dims(img_tta, axis=0)
        output = model.predict(img_tta)[0]
        outputs.append(output)
    if use_avg ==False:
        outputs = np.concatenate(outputs)
    else:
        outputs = np.average(outputs, axis=0)
    print(outputs.shape)
    return outputs
 
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!', os.path.join(dir_name, 'model'))

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!', file_path)

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        query_img = query_img.astype('float32')
        query_img /= 255
        reference_img = reference_img.astype('float32')
        reference_img /= 255

        intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-2].output)
        print('inference start')

        # tta = 8
        # only concate, because we cal distance!
         
        # inference
        query_veclist=[]
        for img in query_img:
            output = predict_tta(intermediate_layer_model,img,8,use_avg=True)
            query_veclist.append(output) 
        query_vecs = np.array(query_veclist)

        # caching db output, db inference
        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            reference_veclist=[]
            print('reference',reference_img.shape)
            for img in reference_img:
                output = predict_tta(intermediate_layer_model,img,8,use_avg=True)
                reference_veclist.append(output) 
            reference_vecs = np.array(reference_veclist)

            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        print(query_vecs.shape, reference_vecs.shape)
        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img

def build_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet', num_classes=1000, base_freeze=True):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predict = Dense(num_classes, activation='softmax', name='last_softmax')(x)
    model = Model(inputs=base_model.input, outputs=predict)
    if base_freeze==True:
        for layer in base_model.layers:
            layer.trainable = False
    return model

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=100)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    NUM_GPU = 1
    SEL_CONF = 2
    CV_NUM = 2

    CONF_LIST = []
    CONF_LIST.append({'name':'Xception', 'input_shape':(224, 224, 3), 'backbone':Xception
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15, 'fc_train_epoch':10})
    CONF_LIST.append({'name':'Resnet50', 'input_shape':(224, 224, 3), 'backbone':ResNet50
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15, 'fc_train_epoch':10})
    CONF_LIST.append({'name':'InceptionResnet', 'input_shape':(224, 224, 3), 'backbone':InceptionResNetV2
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15, 'fc_train_epoch':10})
    CONF_LIST.append({'name':'NASNetLarge', 'input_shape':(224, 224, 3), 'backbone':NASNetLarge
                    , 'batch_size':48 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15, 'fc_train_epoch':10})
    CONF_LIST.append({'name':'DenseNet121', 'input_shape':(224, 224, 3), 'backbone':DenseNet121
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15, 'fc_train_epoch':10})
    CONF_LIST.append({'name':'DenseNet201', 'input_shape':(224, 224, 3), 'backbone':DenseNet201
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15, 'fc_train_epoch':10})


    # training parameters
    nb_epoch = config.epochs
    fc_train_epoch = CONF_LIST[SEL_CONF]['fc_train_epoch']
    fc_train_batch = CONF_LIST[SEL_CONF]['fc_train_batch']
    fc_train_batch *= NUM_GPU
    start_lr = CONF_LIST[SEL_CONF]['start_lr']
    batch_size = CONF_LIST[SEL_CONF]['batch_size']
    batch_size *= NUM_GPU
    num_classes = 1000
    input_shape = CONF_LIST[SEL_CONF]['input_shape']
    SEED = CONF_LIST[SEL_CONF]['SEED']
    backbone = CONF_LIST[SEL_CONF]['backbone']

    """ Model """
    model = build_model(backbone= backbone, use_imagenet=None,input_shape = input_shape, num_classes=num_classes, base_freeze = True)
    model.summary()
    bind_model(model)
    print(model.layers[-2].output.shape)
    if config.pause:
        nsml.paused(scope=locals())
    if config.mode == 'train':
        bTrainmode = True
        #nsml.load(checkpoint='86', session='Zonber/ir_ph1_v2/204') #Nasnet Large 222
        nsml.load(checkpoint='84', session='Zonber/ir_ph1_v2/188') #InceptionResnetV2 222
        nsml.save(0)  # this is display model name at lb