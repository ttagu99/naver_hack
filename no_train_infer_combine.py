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
from classification_models.senet import SEResNeXt50
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
pca_w = './Wpca.npy'
pca_xm = './Xmpca.npy'

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

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db, (299,299))

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
    norm = np.linalg.norm(v,axis=1)
    #if norm == 0:
    #    return v
    return v / norm[:,None]



def extractHueHistogram(img, num_pixels):
    #img = cv2.imread(path, cv2.IMREAD_COLOR)
    #img = cv2.resize(path, size)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = img[:,:,0] ###range: 0-180 (to fit range 0, 255)
    histogram_hue = np.zeros(180, dtype=np.float)
    histogram_hue = np.bincount(hue.ravel(), minlength=180).astype(np.float)
    histogram_hue /= num_pixels
    return histogram_hue

import time

"""
def generator():
    while 1:
        for start in range(0, total, batch_size):
            x_batch = []
            end = min(start + batch_size, total)
            y_batch = y_train[start:end, :]
            file_names_batch = file_names_train[start:end]
            for file_name in file_names_batch:
                im = io.imread(os.path.join(train_jpg, file_name))
                img = im.astype('float32') / 255
                x_batch.append(img)
            yield np.array(x_batch), y_batch
"""

#from skimage.transform import resize
import skimage.transform as trans

"""
def testGenerator(test_path, num_image=30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
"""

# data preprocess
def get_feature(model, queries, db, img_size):
#    img_size = (224, 224)


    batch_size = 200
    #topResultsQE=5
    L=3
    test_path = DATASET_PATH + '/test/test_data'
    intermediate_layer_model_small = Model(inputs=model.get_layer('input_1').input, outputs=[model.get_layer('GAP_LAST').input, model.get_layer('GAP_LAST').output])
    intermediate_layer_model_big = Model(inputs=model.get_layer('input_1_2').input, outputs=[model.get_layer('GAP_LAST_2').input, model.get_layer('GAP_LAST_2').output])
    #intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('GAP_LAST').input)
    
    test_datagen = ImageDataGenerator(rescale=1./255.0, dtype='float32', samplewise_center=True, samplewise_std_normalization=True)
    
    """
    test_generator = test_datagen.flow_from_directory(directory=test_path, target_size=(224, 224), classes=['query'], color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    for t in range(len(test_generator)):
        imgs = heuhist_generator.next()
        for img in imgs:
            img = img.reshape(hue_size[0], hue_size[1], 3)
            img = img.astype(np.uint8)
            query_hue_histogram.append(extractHueHistogram(img, hue_pixels))
        yeild 
    #test_generator2 = testGenerator(test_path)
    """


    
    test_generator_small = test_datagen.flow_from_directory(directory=test_path, target_size=(224, 224), classes=['query'], color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    test_generator_big = test_datagen.flow_from_directory(directory=test_path, target_size=(299, 299), classes=['query'], color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    query_vecs_small, gap_query_vecs_small = intermediate_layer_model_small.predict_generator(test_generator_small, steps=len(test_generator_small), workers=8)
    query_vecs_big, gap_query_big = intermediate_layer_model_big.predict_generator(test_generator_big, steps=len(test_generator_big), workers=8)
    print(query_vecs_small.shape) ######## 8*8*1536 , 6*6*1536 정도로 나올수 있음. 
    print(gap_query_vecs_small.shape)
    print(query_vecs_big.shape)
    print(gap_query_big.shape)

    """
    test_generator_small = test_datagen.flow_from_directory(directory=test_path, target_size=(224, 224), classes=['reference'], color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    test_generator_big = test_datagen.flow_from_directory(directory=test_path, target_size=(299, 299), classes=['reference'], color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    ref_vecs_small, gap_ref_vecs_small = intermediate_layer_model_small.predict_generator([test_generator_small, test_generator_big], steps=len(test_generator_big), workers=8)
    ref_vecs_big, gap_ref_vecs_big = intermediate_layer_model_big.predict_generator([test_generator_small, test_generator_big], steps=len(test_generator_big), workers=8)
    
    print(ref_vecs_small.shape)
    print(gap_ref_vecs_small.shape)
    print(ref_vecs_big.shape)
    print(gap_ref_vecs_big.shape)
    """
    queries, queryMAC, db, DbMAC = []
    """

    #reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),workers=4)

    # ------------------ DB images: reading, descripting and whitening -----------------------
    t1 = time.clock()
    DbMAC = extractRMAC(reference_vecs, intermediate_layer_model, True, L)
    ## vector PCA
    #print('start pca_model_loader')
    #W, Xm = learningPCA(DbMAC)
    #print('W shape:', W.shape, 'Xm shape:',Xm.shape)

    #print("PCA-whitening")
    #DbMAC = apply_whitening(DbMAC, Xm, W)
    DbMAC = np.array(DbMAC)
    DbMAC = sumPooling(DbMAC, reference_vecs.shape[0], False)
    Dbtime = time.clock() - t1
    print('DbMAC lenght',len(DbMAC))
    print("RMAC and PCA-whitening of terminated in",round(Dbtime),"s")

    # ------------------- query images: reading, descripting and whitening -----------------------
    t1 = time.clock()
    queryMAC = extractRMAC(query_vecs, intermediate_layer_model, True, L)
    #print("PCA-whitening")
    #queryMAC = apply_whitening(queryMAC, Xm, W)
    queryMAC = np.array(queryMAC)
    queryMAC = sumPooling(queryMAC, query_vecs.shape[0], False)
    print('queryMAC lenght',len(queryMAC))
    Dbtime = time.clock() - t1
    print("Query descriptors saved!")
    print("queryMAC and PCA-whitening of terminated in",round(Dbtime),"s")

    queryMAC = np.array(queryMAC)
    DbMAC = np.array(DbMAC)
    queryMAC = queryMAC.squeeze()
    DbMAC = DbMAC.squeeze()

    #retrieval1 = time.clock()
    #finalReRank = retrieveQENEW(queryMAC, regions, topResultsQE,url, queryImages, DbImages, dataset)
    #retrieval2 = time.clock() - retrieval1
    #print("AVG query time:",round(retrieval2/len(queryImages),2),"s")

    #retrieval1 = time.clock()
    #finalReRank2 = retrieveQERegionsNEW(queryMAC, regions, topResultsQE, url,queryImages, DbImages, finalReRank, dataset)
    #retrieval2 = time.clock() - retrieval1
    #print("AVG query expansion time:",round(retrieval2/len(queryImages),2),"s")
    """
    return queries, queryMAC, db, DbMAC
    
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
    #model = build_model(backbone= InceptionResNetV2, input_shape = input_shape, use_imagenet = None, num_classes=num_classes, base_freeze=True, opt =opt,use_gap_net=use_gap_net)
    #bind_model(model)

    small_model = build_model(backbone= InceptionResNetV2, input_shape = (224, 224, 3), use_imagenet = None, num_classes=num_classes, base_freeze=True, opt =opt,use_gap_net=use_gap_net)
    #intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('GAP_LAST').input)
    x = small_model.output
    fake_out_small = Dense(num_classes, activation='softmax', name='last_softmax1')(x)
    
    #bind_model(small_model)
    #nsml.load(checkpoint='secls_637_19', session='Zonber/ir_ph2/466') #InceptionResnetV2 222
    small_intermediate_layer_model = Model(inputs=small_model.input, outputs=fake_out_small)

    #small_intermediate_layer_model = Model(inputs=small_model.input, outputs=[small_model.get_layer('GAP_LAST').input, small_model.get_layer('GAP_LAST').output])
    #small_intermediate_layer_model.add
    
    large_model = build_model(backbone= InceptionResNetV2, input_shape = (299, 299, 3), use_imagenet = None, num_classes=num_classes, base_freeze=True, opt =opt,use_gap_net=use_gap_net)
    x = large_model.output
    for i, layer in enumerate(large_model.layers):
        layer.name += '_2'

    fake_out_big = Dense(num_classes, activation='softmax', name='last_softmax2')(x)
    #bind_model(large_model)
    #nsml.load(checkpoint='secls_222_33', session='Zonber/ir_ph2/314')
    large_intermediate_layer_model = Model(inputs=large_model.input, outputs=fake_out_big)#[large_model.get_layer('GAP_LAST').input, large_model.get_layer('GAP_LAST').output])

    finalOut = keras.layers.concatenate([small_intermediate_layer_model.output,large_intermediate_layer_model.output], axis=1) #Concatenate(axis=1)([small_intermediate_layer_model.output,large_intermediate_layer_model.output])
    joinedModel = Model([small_intermediate_layer_model.input, large_intermediate_layer_model.input], finalOut)
    bind_model(joinedModel)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

    nsml.load(checkpoint='over_over_fitting_comb', session='Zonber/ir_ph2/618') 
    nsml.save('over_over_fitting_comb')

    #nsml.load(checkpoint='secls_222_27', session='Zonber/ir_ph2/314') #InceptionResnetV2 222
    #nsml.save('over_over_fitting')  # this is display model name at lb