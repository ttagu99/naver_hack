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

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        #model.save(os.path.join(dir_name, 'model'))
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!', os.path.join(dir_name, 'model'))

    def load(file_path):
        model.load_weights(file_path)
        #model = load_model(file_path)
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

        # inference
        query_veclist=[]
        for img in query_img:
            img = np.expand_dims(img, axis=0)
            print(img.shape)
            output = intermediate_layer_model.predict(img)[0]
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
                img = np.expand_dims(img, axis=0)
                print(img.shape)
                output = intermediate_layer_model.predict(img)[0]
                reference_veclist.append(output) 
            reference_vecs = np.array(reference_veclist)
            print('reference_vecs', reference_vecs.shape)

            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

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


#why generator make low score when train by multi gpu  
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, features, labels, batch_size, aug_seq, num_classes, use_aug = True):
        'Initialization'
        self.features = features
        self.batch_size = batch_size
        self.labels = labels
        self.aug_seq = aug_seq
        self.use_aug = use_aug
        self.num_classes = num_classes

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_features = np.zeros((self.batch_size, self.features.shape[1], self.features.shape[2], self.features.shape[3]))
        batch_labels = np.zeros((self.batch_size, self.num_classes))
        indexes = random.sample(range(len(self.features)), self.batch_size)

        if self.use_aug == True:
            batch_features[:,:,:,:]  = self.aug_seq.augment_images(self.features[indexes])
        else:
            batch_features[:,:,:,:]  = self.features[indexes]
        batch_labels[:,:] =  self.labels[indexes]
        batch_features = batch_features.astype('float32')
        batch_features /= 255
        return batch_features, batch_labels

def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')    
    return model

class report_nsml(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, epoch=epoch, loss=logs.get('loss'), acc=logs.get('acc'), val_loss=logs.get('val_loss'), val_acc=logs.get('val_acc'))
        nsml.save(epoch)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=100)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--g', type=int, default=0, help='gpu')
    config = args.parse_args()

    NUM_GPU = 1
    SEL_CONF = 3
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
    model = build_model(backbone= backbone, use_imagenet='imagenet',input_shape = input_shape, num_classes=num_classes, base_freeze = True)
    model.summary()
    bind_model(model)
    print(model.layers[-2].output.shape)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        opt = keras.optimizers.Adam(lr=start_lr)
        if NUM_GPU != 1:
            model = keras.utils.multi_gpu_model(model, gpus=NUM_GPU)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
                       output_path=output_path)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
            train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)

        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)

        x_train = np.asarray(img_list)
        labels = np.asarray(label_list)
        y_train = keras.utils.to_categorical(labels, num_classes=num_classes)

        print(len(labels), 'train samples')
        mean_ch = x_train.mean()
        std_ch = x_train.std()
        print('mean:',mean_ch, 'std:',std_ch)

        xx_train, xx_val, yy_train, yy_val = train_test_split(x_train, y_train, test_size=0.15, random_state=SEED,stratify=y_train)
        xx_val = xx_val.astype('float32')
        xx_val /= 255
        print('shape:',xx_train.shape,'train max v:',xx_train.max(), 'train mean v:' , xx_train.mean())
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 3),[
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=['reflect']
                )),
                sometimes( iaa.OneOf([
                    iaa.Affine(rotate=0),
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270)
                ])),
                sometimes(iaa.Affine(
                    scale={"x": (0.1, 1.1), "y": (0.9, 1.1)}, 
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, 
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-5, 5), 
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    mode=['reflect'] 
                ))
                ]),
            ],
            random_order=True
        )



        """ Callback """
        monitor = 'val_acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=9,factor=0.2,verbose=1)
        early_stop = EarlyStopping(monitor=monitor, patience=20)
        best_model_path = './best_model.h5'
        checkpoint = ModelCheckpoint(best_model_path,monitor=monitor,verbose=1,save_best_only=True)
        report = report_nsml()
        callbacks = [reduce_lr,early_stop,checkpoint,report]

        #nsml.load(checkpoint='70', session='Zonber/ir_ph1_v2/163')
        train_gen = DataGenerator(xx_train, yy_train,fc_train_batch,seq,num_classes,use_aug=True)
        #train_gen = generator(xx_train, yy_train,fc_train_batch,seq,num_classes,use_aug=True)
        hist1 = model.fit_generator(train_gen,validation_data= (xx_val,yy_val), workers=8, use_multiprocessing=True
                ,  epochs=fc_train_epoch,  callbacks=callbacks,   verbose=1, shuffle=True)

        for layer in model.layers:
            layer.trainable=True
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy']) 
        
        #model.load_weights(best_model_path)
        print('load model:' ,best_model_path)

        train_gen = DataGenerator(xx_train, yy_train,batch_size,seq,num_classes,use_aug=True)
        #train_gen = generator(xx_train, yy_train,batch_size,seq,num_classes,use_aug=True)
        hist2 = model.fit_generator(train_gen ,validation_data= (xx_val,yy_val), workers=8, use_multiprocessing=True
                 ,  epochs=nb_epoch,  callbacks=callbacks,   verbose=1, shuffle=True)

        best_epoch = np.argmax(hist2.history[monitor]).astype(int)
        train_loss, train_acc= hist2.history['loss'][best_epoch], hist2.history['acc'][best_epoch]
        val_loss, val_acc=  hist2.history['val_loss'][best_epoch], hist2.history['val_acc'][best_epoch]