# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,Input, GlobalMaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import backend as K
from data_loader import train_data_loader, pca_model_loader
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from classification_models.resnet import ResNet18, SEResNet18
from classification_models.senet import SEResNeXt50,SEResNeXt101
from keras.models import Model,load_model
from keras.optimizers import Adam, SGD
from keras import Model, Input
from keras.layers import Layer, multiply, Lambda
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa
import random
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf
from keras.losses import categorical_crossentropy
from utils import *
import pickle
from sklearn.decomposition import PCA

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs, indices = get_feature(model, queries, db, (299,299))



        # Calculate cosine similarity
        #sim_matrix = np.dot(query_vecs, reference_vecs.T)
        #indices = np.argsort(sim_matrix, axis=1)
        #indices = np.flip(indices, axis=1)

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
    norm = np.linalg.norm(v,axis=1)
    #if norm == 0:
    #    return v
    return v / norm[:,None]

def extractHueHistogram(img):
   #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   hue = img[:,:,0] ###range: 0-180 (to fit range 0, 255)
   histogram_hue = np.zeros(180, dtype=np.float)
   histogram_hue = np.bincount(hue.ravel(), minlength=180).astype(np.float)
   num_pixels = (img.shape[0] * img.shape[1])
   histogram_hue /= num_pixels
   return histogram_hue

def docom_feature(query_vecs,reference_vecs,gap_query_vecs,gap_reference_vecs):
    L=3
    # ------------------ DB images: reading, descripting and whitening -----------------------
    DbMAC = extractRMAC(reference_vecs, True, L)
    DbMAC = np.array(DbMAC)
    DbMAC_sumpool = sumPooling(DbMAC, reference_vecs.shape[0], False)
    print('DbMAC_sumpool lenght',len(DbMAC_sumpool))

    # ------------------- query images: reading, descripting and whitening -----------------------
    queryMAC = extractRMAC(query_vecs, True, L)
    queryMAC = np.array(queryMAC)
    queryMAC_sumpool = sumPooling(queryMAC, query_vecs.shape[0], False)
    print('queryMAC_sumpool lenght',len(queryMAC_sumpool))

    DbMAC_sumpool = np.array(DbMAC_sumpool)
    DbMAC_sumpool = DbMAC_sumpool.squeeze()
    queryMAC_sumpool = np.array(queryMAC_sumpool)
    queryMAC_sumpool = queryMAC_sumpool.squeeze()

######################################
 #   queryMAC = queryMAC.squeeze()
 #   DbMAC = DbMAC.squeeze()
 #   print('DbMAC.shape',DbMAC.shape)
	### query regions - db regions l2_nor
 #   queryMAC = l2_normalize(queryMAC)
 #   DbMAC = l2_normalize(DbMAC)
 #   region_number = queryMAC.shape[0]//query_vecs.shape[0]
 #   print('DbMAC.shape',DbMAC.shape, 'region number', region_number)
 #   query_con_rmac = queryMAC.reshape((queryMAC.shape[0]//region_number, queryMAC.shape[1]*region_number))
 #   db_con_rmac = DbMAC.reshape((DbMAC.shape[0]//region_number, DbMAC.shape[1]*region_number))
 #   print('query_con_rmac.shape',query_con_rmac.shape,'db_con_rmac.shape',db_con_rmac.shape)
 #    # pca decom_l2
 #   all_vecs = np.concatenate([query_con_rmac, db_con_rmac])
 #   print('pca rmac all')
 #   all_pca_vecs = PCA(256).fit_transform(all_vecs)
 #   print('pca rmac all l2')
 #   all_pca_vecs = l2_normalize(all_pca_vecs)
 #   print('query_con_rmac')
 #   query_con_rmac = all_pca_vecs[:query_con_rmac.shape[0],:]
 #   print('db_con_rmac')
 #   db_con_rmac = all_pca_vecs[query_con_rmac.shape[0]:,:]
 #   print('reshape concate per image', db_con_rmac.shape)
 #######################################

##   # query regions - db regions simimlarity
#     # pca decom_l2
#    all_vecs = np.concatenate([queryMAC, DbMAC])
#    print('pca rmac all')
#    all_pca_vecs = PCA(128).fit_transform(all_vecs)
#    all_pca_vecs = l2_normalize(all_pca_vecs)
#    queryMAC = all_pca_vecs[:queryMAC.shape[0],:]
#    DbMAC = all_pca_vecs[queryMAC.shape[0]:,:]
#    region_number = queryMAC.shape[0]//query_vecs.shape[0]
#    print('DbMAC.shape',DbMAC.shape, 'region number', region_number)
#    #concate
#    query_con_rmac = queryMAC.reshape((queryMAC.shape[0]//region_number, queryMAC.shape[1]*region_number))
#    db_con_rmac = DbMAC.reshape((DbMAC.shape[0]//region_number, DbMAC.shape[1]*region_number))
#    query_con_rmac = l2_normalize(query_con_rmac)
#    db_con_rmac = l2_normalize(db_con_rmac)
#    print('reshape concate per image', db_con_rmac.shape)

    # pca decom_l2
    #all_vecs = np.concatenate([query_con_rmac, db_con_rmac])
    #print('pca rmac all second')
    #all_pca_vecs = PCA(256).fit_transform(all_vecs)
    #all_pca_vecs = l2_normalize(all_pca_vecs)
    #query_con_rmac = all_pca_vecs[:query_con_rmac.shape[0],:]
    #db_con_rmac = all_pca_vecs[query_con_rmac.shape[0]:,:]
######################################
    
    # l2
    #queryMAC_sumpool = l2_normalize(queryMAC_sumpool)
    #DbMAC_sumpool = l2_normalize(DbMAC_sumpool)
    gap_query_vecs = l2_normalize(gap_query_vecs)
    gap_reference_vecs = l2_normalize(gap_reference_vecs)

    query_vecs = np.concatenate([queryMAC_sumpool,gap_query_vecs],axis=1)
    reference_vecs = np.concatenate([DbMAC_sumpool, gap_reference_vecs],axis=1)

    # l2 normalization
    query_vecs = l2_normalize(query_vecs)
    reference_vecs = l2_normalize(reference_vecs)

    # pca
    all_vecs = np.concatenate([query_vecs, reference_vecs])
    all_pca_vecs = PCA(1024).fit_transform(all_vecs)
    query_vecs = all_pca_vecs[:query_vecs.shape[0],:]
    reference_vecs = all_pca_vecs[query_vecs.shape[0]:,:]

    # l2 normalization
    query_vecs = l2_normalize(query_vecs)
    reference_vecs = l2_normalize(reference_vecs)

    # Calculate cosine similarity for QE
    qe_iter = 2
    qe_number = 19
    dba_number = 9
    for i in range(qe_iter):
        qe_number = qe_number // (i+1)
        dba_number = dba_number // (i+1)
        weights = np.logspace(0, -1.5, (qe_number+1))
        weights /= weights.sum()
        pre_sim_matrix = np.dot(query_vecs, reference_vecs.T)
        pre_indices = np.argsort(pre_sim_matrix, axis=1) #lower first
        pre_indices = np.flip(pre_indices, axis=1) #higher first
        for i in range(query_vecs.shape[0]):
            query_vecs[i] *= weights[0]
            for refidx in range(qe_number):
                query_vecs[i] += reference_vecs[pre_indices[i][refidx]]*weights[refidx+1]

        # after query expanstion l2 normalization
        query_vecs = l2_normalize(query_vecs)

        # Calculate cosine similarity for DBA
        weights = np.logspace(0, -1.5, (dba_number+1))
        weights /= weights.sum()
        pre_sim_matrix = np.dot(reference_vecs, query_vecs.T)
        pre_indices = np.argsort(pre_sim_matrix, axis=1) #lower first
        pre_indices = np.flip(pre_indices, axis=1) #higher first
        for i in range(reference_vecs.shape[0]):
            reference_vecs[i] *= weights[0]
            for refidx in range(dba_number):
                reference_vecs[i] += query_vecs[pre_indices[i][refidx]]*weights[refidx+1]

        # after database augment l2 normalization
        reference_vecs = l2_normalize(reference_vecs)
        
    return query_vecs, reference_vecs

# data preprocess
def get_feature(model, queries, db, img_size):
#    img_size = (224, 224)


    batch_size = 200
    #topResultsQE=5
    test_path = DATASET_PATH + '/test/test_data'
    intermediate_layer_model = Model(inputs=model.input, outputs=[model.get_layer('GAP_LAST').input,model.get_layer('GAP_LAST').output])
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32', samplewise_center=True, samplewise_std_normalization=True)
    test_datagen_lr = ImageDataGenerator(rescale=1. / 255, dtype='float32', samplewise_center=True, samplewise_std_normalization=True,preprocessing_function = np.fliplr)    
    #huehist_gen = ImageDataGenerator()
    #huehist_generator = huehist_gen.flow_from_directory(
    #    directory=test_path,
    #    target_size=img_size,
    #    classes=['query'],
    #    color_mode="rgb",
    #    batch_size=1,
    #    class_mode=None,
    #    shuffle=False
    #)

    #for i in range(huehist_gen)


    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=img_size,
        classes=['query'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    query_generator_lr = test_datagen_lr.flow_from_directory(
        directory=test_path,
        target_size=img_size,
        classes=['query'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )    


    query_vecs, gap_query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator),workers=4)
    query_vecs_lr,gap_query_vecs_lr = intermediate_layer_model.predict_generator(query_generator_lr, steps=len(query_generator_lr),workers=4)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=img_size,
        classes=['reference'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    reference_generator_lr = test_datagen_lr.flow_from_directory(
        directory=test_path,
        target_size=img_size,
        classes=['reference'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    reference_vecs, gap_reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),workers=4)
    reference_vecs_lr,gap_reference_vecs_lr = intermediate_layer_model.predict_generator(reference_generator_lr, steps=len(reference_generator_lr),workers=4)

    
    query_vecs, reference_vecs = docom_feature(query_vecs,reference_vecs,gap_query_vecs,gap_reference_vecs)
    query_vecs_lr, reference_vecs_lr = docom_feature(query_vecs_lr,reference_vecs_lr,gap_query_vecs_lr,gap_reference_vecs_lr)
    query_vecs = np.concatenate([query_vecs,query_vecs_lr],axis=1)
    reference_vecs = np.concatenate([reference_vecs,reference_vecs_lr],axis=1)

    query_vecs =l2_normalize(query_vecs)
    reference_vecs = l2_normalize(reference_vecs)

    # LAST Calculate cosine similarity
    qe_sim_matrix = np.dot(query_vecs, reference_vecs.T)
    qe_indices = np.argsort(qe_sim_matrix, axis=1)
    qe_indices = np.flip(qe_indices, axis=1)

    return queries, query_vecs, db, reference_vecs, qe_indices

def build_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet', num_classes=1383, base_freeze=True, opt = SGD(), NUM_GPU=1,use_gap_net=False):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    #x = Flatten(name='FLATTEN_LAST')(x)


    if use_gap_net ==True:
        #skip_connection_layers = (594, 260, 16, 9)
        gap1 = GlobalAveragePooling2D(name='GAP_1')(base_model.layers[594].output)
        gap2 = GlobalAveragePooling2D(name='GAP_2')(base_model.layers[260].output)
        gap3 = GlobalAveragePooling2D(name='GAP_3')(base_model.layers[16].output)
        gap4 = GlobalAveragePooling2D(name='GAP_4')(base_model.layers[9].output)
        #gmp = GlobalMaxPooling2D(name='GMP_LAST')(x)
        gap = GlobalAveragePooling2D(name='GAP_0')(x)
        g_con = Concatenate(name='GAP_LAST')([gap,gap1,gap2,gap3,gap4])
        g_con = Dropout(rate=0.5)(g_con)
    else:
        gap = GlobalAveragePooling2D(name='GAP_LAST')(x)
        g_con = Dropout(rate=0.5)(gap)

    predict = Dense(num_classes, activation='softmax', name='last_softmax')(g_con)
    model = Model(inputs=base_model.input, outputs=predict)
    if base_freeze==True:
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(loss='categorical_crossentropy',   optimizer=opt,  metrics=['accuracy'])
    return model


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_paths, input_shape, labels, batch_size, aug_seq, num_classes, use_aug = True, shuffle = True, mean=None):
        'Initialization'
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.labels = labels
        self.aug_seq = aug_seq
        self.use_aug = use_aug
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.mean = mean

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_features = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        features = []
        batch_labels = np.zeros((self.batch_size, self.num_classes))
        indexes = self.indexes[index*self.batch_size:(index+1)* self.batch_size]
        files = self.image_paths[indexes]
        features = read_image_batch(files, (self.input_shape[0], self.input_shape[1]))
        features = np.array(features)
        if self.use_aug == True:
            batch_features[:,:,:,:]  = self.aug_seq.augment_images(features)
        else:
            batch_features[:,:,:,:]  = features

        batch_labels[:,:] =  self.labels[indexes]
        batch_features = normal_inputs(batch_features,self.mean)
        return batch_features, batch_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        org_idx = np.arange(len(self.image_paths))
        mod_idx = np.random.choice(org_idx, (self.__len__()*self.batch_size) - len(self.image_paths))
        self.indexes = np.concatenate([org_idx,mod_idx])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class report_nsml(keras.callbacks.Callback):
    def __init__(self, prefix, seed):
        'Initialization'
        self.prefix = prefix
        self.seed = seed
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, epoch=epoch, loss=logs.get('loss'), val_loss=logs.get('val_loss'),acc=logs.get('acc'),val_acc=logs.get('val_acc'))
        nsml.save(self.prefix +'_'+ str(self.seed)+'_' +str(epoch))

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--batch_size', type=int, default=60)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    lesssometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential(
        [
            iaa.SomeOf((0, 3),[
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            sometimes(iaa.CropAndPad(
                percent=(-0.1, 0.2),
                pad_mode=['reflect']
            )),
            sometimes( iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)
            ])),
            sometimes(iaa.Affine(
                scale={"x": (0.7, 1.3), "y": (0.7, 1.3)}, 
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, 
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-5, 5), 
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                mode=['reflect'] 
            )),
            lesssometimes( iaa.SomeOf((0, 5),
                        [
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)),
                                iaa.AverageBlur(k=(2, 7)),
                                iaa.MedianBlur(k=(3, 5)),
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                            sometimes(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0, 0.7)),
                                iaa.DirectedEdgeDetect(
                                    alpha=(0, 0.7), direction=(0.0, 1.0)
                                ),
                            ])),
                            iaa.AdditiveGaussianNoise(
                                loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                            ),
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                iaa.CoarseDropout(
                                    (0.03, 0.15), size_percent=(0.02, 0.05),
                                    per_channel=0.2
                                ),
                            ]),
                            iaa.Invert(0.05, per_channel=True), # invert color channels
                            iaa.Add((-10, 10), per_channel=0.5),
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            sometimes(
                                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                            ),
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                        ],
                        random_order=True
                    )),
            ]),
        ],
        random_order=True
    )
    
    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (299,299,3)#(224, 224, 3)  # input image shape
    use_gap_net = False
    opt = keras.optimizers.Adam(lr=0.0005)
    model = build_model(backbone= InceptionResNetV2, input_shape = input_shape, use_imagenet = None, num_classes=num_classes, base_freeze=True, opt =opt,use_gap_net=use_gap_net)
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

    nsml.load(checkpoint='secls_222_27', session='Zonber/ir_ph2/314') #InceptionResnetV2 222
    nsml.save('over_over_fitting')  # this is display model name at lb
    #for i in range(14,23):
    #    nsml.load(checkpoint='secls_222_'+str(i), session='Zonber/ir_ph2/450')
    #    nsml.save('secls_222_'+str(i))
