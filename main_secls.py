# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import cv2
import pickle
import time
import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential
from keras.layers import Concatenate
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
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model,load_model
from keras.optimizers import Adam, SGD
from keras import Model, Input
from keras.layers import Layer

from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa
import random
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):
        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

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
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('features').output)
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs

def build_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet', num_classes=1383, base_freeze=True, opt = SGD(), NUM_GPU=1):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    x = GlobalAveragePooling2D(name='features')(x)
    predict = Dense(num_classes, activation='softmax', name='last_softmax')(x)
    model = Model(inputs=base_model.input, outputs=predict)
    if base_freeze==True:
        for layer in base_model.layers:
            layer.trainable = False

    if NUM_GPU != 1:
        model = keras.utils.multi_gpu_model(model, gpus=NUM_GPU)
    model.compile(loss='categorical_crossentropy',   optimizer=opt,  metrics=['accuracy'])
    return model

class report_nsml(keras.callbacks.Callback):
    def __init__(self, prefix, seed):
        'Initialization'
        self.prefix = prefix
        self.seed = seed
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, epoch=epoch, loss=logs.get('loss'), acc=logs.get('acc'), val_loss=logs.get('val_loss'), val_acc=logs.get('val_acc'))
        #nsml.save(self.prefix +'_'+ str(self.seed)+'_' +str(epoch))

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
    CV_NUM = 1

    CONF_LIST = []
    CONF_LIST.append({'name':'Xc', 'input_shape':(224, 224, 3), 'backbone':Xception
                    , 'batch_size':100 ,'fc_train_batch':330, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'Re50', 'input_shape':(224, 224, 3), 'backbone':ResNet50
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'TTM', 'input_shape':(224, 224, 3), 'backbone':InceptionResNetV2
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':1, 'fc_train_epoch':1, 'imagenet':None})
    CONF_LIST.append({'name':'IR', 'input_shape':(224, 224, 3), 'backbone':InceptionResNetV2
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':100, 'fc_train_epoch':3, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'NL', 'input_shape':(224, 224, 3), 'backbone':NASNetLarge
                    , 'batch_size':48 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'De121', 'input_shape':(224, 224, 3), 'backbone':DenseNet121
                    , 'batch_size':256 ,'fc_train_batch':520, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':20, 'fc_train_epoch':2, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'De201', 'input_shape':(224, 224, 3), 'backbone':DenseNet201
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})

    # training parameters
    use_merge_bind = False
    nb_epoch = CONF_LIST[SEL_CONF]['epoch']
    fc_train_epoch = CONF_LIST[SEL_CONF]['fc_train_epoch']
    fc_train_batch = CONF_LIST[SEL_CONF]['fc_train_batch']
    fc_train_batch *= NUM_GPU
    start_lr = CONF_LIST[SEL_CONF]['start_lr']
    batch_size = CONF_LIST[SEL_CONF]['batch_size']
    batch_size *= NUM_GPU
    num_classes = 1383
    input_shape = CONF_LIST[SEL_CONF]['input_shape']
    SEED = CONF_LIST[SEL_CONF]['SEED']
    backbone = CONF_LIST[SEL_CONF]['backbone']
    prefix = CONF_LIST[SEL_CONF]['name']
    use_imagenet = CONF_LIST[SEL_CONF]['imagenet']

    """ CV Model """
    opt = keras.optimizers.Adam(lr=start_lr)
    if use_merge_bind == True:
        feature_models = []
        for cv in range(CV_NUM):
            temp_model = build_model(backbone= backbone, use_imagenet=None,input_shape = input_shape, num_classes=num_classes, base_freeze = True,opt = opt)
            feature_model = Model(inputs=temp_model.inputs,outputs = temp_model.layers[-2].output)
            feature_models.append(feature_model)
        model_input = Input(shape=input_shape)
        en_model = ensemble_feature_vec(feature_models,model_input, num_classes)
        bind_model(en_model)
        en_model.summary()
    else:
        model = build_model(backbone= backbone, use_imagenet=None,input_shape = input_shape, num_classes=num_classes, base_freeze = True,opt = opt)
        bind_model(model)
        model.summary()

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

    mean_arr = None# np.zeros(input_shape)
    #for img in img_list:
    #    mean_arr += img.astype('float32')
    #mean_arr /= len(img_list)
    #print('mean shape:',mean_arr.shape, 'mean mean:',mean_arr.mean(), 'mean max:',mean_arr.max())
    #mean_arr /= 255
    #np.save('./mean.npy', mean_arr)


    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        x_train = np.asarray(img_list)
        labels = np.asarray(label_list)
        y_train = keras.utils.to_categorical(labels, num_classes=num_classes)

        print(len(labels), 'train samples')

        best_model_paths = []
        for cv in range(CV_NUM):
            cur_seed = SEED + cv
            opt = keras.optimizers.Adam(lr=start_lr)
            model = build_model(backbone= backbone, use_imagenet=use_imagenet,input_shape = input_shape, num_classes=num_classes, base_freeze = True,opt = opt, NUM_GPU=NUM_GPU)
            #xx_train, xx_val, yy_train, yy_val = train_test_split(x_train, y_train, test_size=0.15, random_state=cur_seed,stratify=y_train)
            #print('shape:',xx_train.shape,'val shape:',xx_val.shape)
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
            reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=4,factor=0.2,verbose=1)
            early_stop = EarlyStopping(monitor=monitor, patience=7)
            best_model_path = './best_model' + str(cur_seed) + '.h5'
            best_model_paths.append(best_model_path)
            checkpoint = ModelCheckpoint(best_model_path,monitor=monitor,verbose=1,save_best_only=True)
            report = report_nsml(prefix = prefix,seed = cur_seed)
            callbacks = [reduce_lr,early_stop,checkpoint,report]

            datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2
                    ,preprocessing_function=seq.augment_image)

            train_generator = datagen.flow_from_directory(
                directory=DATASET_PATH + '/train/train_data',
                target_size=input_shape[:2],
                color_mode="rgb",
                batch_size=fc_train_batch,
                class_mode="categorical",
                shuffle=True,
                seed=SEED,
                subset = 'training'
            )

            val_generator  = datagen.flow_from_directory(
                directory=DATASET_PATH + '/train/train_data',
                target_size=input_shape[:2],
                color_mode="rgb",
                batch_size=fc_train_batch,
                class_mode="categorical",
                shuffle=False,
                seed=SEED,
                subset = 'validation'
            )
            STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
            STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

            hist1 = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN
                                        , validation_data=val_generator, validation_steps = STEP_SIZE_VAL,  workers=4, use_multiprocessing=False
                                 ,  epochs=fc_train_epoch,  callbacks=callbacks,   verbose=1, shuffle=True)


            for layer in model.layers:
                layer.trainable=True
            model.compile(loss='categorical_crossentropy',  optimizer=opt,  metrics=['accuracy']) 
        
            model.load_weights(best_model_path)
            print('load model:' ,best_model_path)

            train_generator = datagen.flow_from_directory(
                directory=DATASET_PATH + '/train/train_data',
                target_size=input_shape[:2],
                color_mode="rgb",
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,
                seed=SEED,
                subset = 'training'
            )

            val_generator  = datagen.flow_from_directory(
                directory=DATASET_PATH + '/train/train_data',
                target_size=input_shape[:2],
                color_mode="rgb",
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
                seed=SEED,
                subset = 'validation'
            )

            STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
            STEP_SIZE_VAL = val_generator.n // val_generator.batch_size
            hist2 = model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN
                                        , validation_data=val_generator, validation_steps = STEP_SIZE_VAL,  workers=4, use_multiprocessing=False
                                 ,  epochs=nb_epoch,  callbacks=callbacks,   verbose=1, shuffle=True)

            model.load_weights(best_model_path)
            nsml.save(prefix + str(cv))

        if use_merge_bind == True:
            print('all cv model train complete, now cv model saving start')
            feature_models = []
            for bp in best_model_paths:
                temp_model = load_model(bp)
                feature_model = Model(inputs=temp_model.inputs,outputs = temp_model.layers[-2].output)
                feature_models.append(feature_model)

            model_input = Input(shape=input_shape)
            en_model = ensemble_feature_vec(feature_models,model_input, num_classes)
            en_model.save('./ensemble.h5')
            print('save model:',prefix +'Merge' + str(CV_NUM))
            nsml.report(summary=True)
            nsml.save(prefix +'Merge' + str(CV_NUM))