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
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa
import random
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator


def get_tta_image(image, tta):
    images=[]
    temp = image
    images.append(temp)
    if tta ==2:
        images.append(np.fliplr(temp))
    elif tta ==4:
        images.append(np.fliplr(temp))
        for i in range(len(images)):
            images.append(np.flipud(images[i]))
    elif tta == 8:
        for i in range(3):
            temp = np.rot90(temp)
            images.append(temp)
        for i in range(len(images)):
            images.append(np.fliplr(images[i])) 
    else:
        pass
    return images

def predict_tta_batch(model, imgs, tta=8, batch_size = 100, use_avg=False):
    all_set_num = len(imgs)
    tta_batch_num = batch_size*tta
    outputs  = []
    for start in range(0,all_set_num,tta_batch_num):
        end = start+tta_batch_num
        end = min(end, all_set_num)
        cur_batch_imgs = imgs[start:end]

        cur_batch_img_ttas= []
        for img in cur_batch_imgs:
            cur_batch_img_ttas += get_tta_image(img,tta)
        cur_batch = np.array(cur_batch_img_ttas)
        #print('cur_batch.shape',cur_batch.shape)
        cur_output = model.predict(cur_batch)
        #print('len cur_output',len(cur_output))
        for tta_start in range(0,len(cur_output),tta):
            tta_end = tta_start+tta
            per_one_img_output = cur_output[tta_start:tta_end]
            if use_avg ==False:
                per_one_img_output = np.concatenate(per_one_img_output)
            else:
                per_one_img_output = np.average(per_one_img_output,axis=0)
            outputs.append(per_one_img_output)
    return outputs

def normal_inputs(imgs, mean_arr=None):
    if isinstance(imgs, (list,)):
        results = []
        for img in imgs:
            results.append(normal_input(img,mean_arr))
        return results
    else:
        return normal_input(imgs,mean_arr)

def normal_input(img, mean_arr=None):
    img = img.astype('float32')
    img /= 255
    #img -= mean_arr
    return img
 

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
    norm = np.linalg.norm(v,axis=1)
    #if norm == 0:
    #    return v
    return v / norm[:,None]


# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    batch_size = 200
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('GAP_LAST').output)
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=img_size,
        classes=['query'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=img_size,
        classes=['reference'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs

def build_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet', num_classes=1383, base_freeze=True, opt = SGD(), NUM_GPU=1):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    x = GlobalAveragePooling2D(name='GAP_LAST')(x)
    predict = Dense(num_classes, activation='softmax', name='last_softmax')(x)
    model = Model(inputs=base_model.input, outputs=predict)
    if base_freeze==True:
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(loss='categorical_crossentropy',   optimizer=opt,  metrics=['accuracy'])
    return model

# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    query_img = read_image_batch(queries,img_size)
    reference_img = read_image_batch(db, img_size)
    return queries, query_img, db, reference_img

def build_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet', num_classes=1383, base_freeze=True, opt = SGD(), NUM_GPU=1):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    x = GlobalAveragePooling2D(name='GAP_LAST')(x)
    predict = Dense(num_classes, activation='softmax', name='last_softmax')(x)
    model = Model(inputs=base_model.input, outputs=predict)
    if base_freeze==True:
        for layer in base_model.layers:
            layer.trainable = False

    if NUM_GPU != 1:
        model = keras.utils.multi_gpu_model(model, gpus=NUM_GPU)
    model.compile(loss='categorical_crossentropy',   optimizer=opt,  metrics=['accuracy'])
    return model

def read_image(img_path,shape):
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    return img

def read_image_batch(image_paths, shape):
    images = []
    for img_path in image_paths:
        img = read_image(img_path,shape)
        images.append(img)
    return images

#why generator make low score when train by multi gpu  
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

def ensemble_feature_vec(models, model_input, num_classes):
    yModels=[model(model_input) for model in models] 
    yAvg=Concatenate()(yModels) 
    yAvg = Dense(num_classes, activation='softmax', name='dummy_sf')(yAvg)
    modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  
    return modelEns

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
    SEL_CONF = 2
    CV_NUM = 5

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
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'NL', 'input_shape':(224, 224, 3), 'backbone':NASNetLarge
                    , 'batch_size':48 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'De121', 'input_shape':(224, 224, 3), 'backbone':DenseNet121
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
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
        #nsml.load(checkpoint='86', session='Zonber/ir_ph1_v2/204') #Nasnet Large 222
        nsml.load(checkpoint='10', session='Zonber/ir_ph2/206') #InceptionResnetV2 222
        nsml.save(0)  # this is display model name at lb