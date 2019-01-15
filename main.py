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

def normal_input(img, mean_arr):
    img = img.astype('float32')
    img /= 255
    #img -= mean_arr
    return img
 
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
        mean_arr = np.load('./mean.npy')
        query_img = normal_input(query_img,mean_arr)
        reference_img = normal_input(reference_img,mean_arr)

        intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-2].output)
        print('inference start')

        # inference
        query_veclist=[]
        for img in query_img:
            output = predict_tta(intermediate_layer_model,img,8,use_avg=False)
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
                output = predict_tta(intermediate_layer_model,img,8,use_avg=False)
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

def build_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet', num_classes=1000, base_freeze=True, opt = SGD(), NUM_GPU=1):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predict = Dense(num_classes, activation='softmax', name='last_softmax')(x)
    model = Model(inputs=base_model.input, outputs=predict)
    if base_freeze==True:
        for layer in base_model.layers:
            layer.trainable = False

    if NUM_GPU != 1:
        model = keras.utils.multi_gpu_model(model, gpus=NUM_GPU)
    model.compile(loss='categorical_crossentropy',   optimizer=opt,  metrics=['accuracy'])
    return model


#why generator make low score when train by multi gpu  
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, features, labels, batch_size, aug_seq, num_classes, use_aug = True, shuffle = True, mean=None):
        'Initialization'
        self.features = features
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
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_features = np.zeros((self.batch_size, self.features.shape[1], self.features.shape[2], self.features.shape[3]))
        batch_labels = np.zeros((self.batch_size, self.num_classes))
        indexes = self.indexes[index*self.batch_size:(index+1)* self.batch_size]
        if self.use_aug == True:
            batch_features[:,:,:,:]  = self.aug_seq.augment_images(self.features[indexes])
        else:
            batch_features[:,:,:,:]  = self.features[indexes]
        batch_labels[:,:] =  self.labels[indexes]
        batch_features = normal_input(batch_features,self.mean)
        return batch_features, batch_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        org_idx = np.arange(len(self.features))
        mod_idx = np.random.choice(org_idx, (self.__len__()*self.batch_size) - len(self.features))
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
    SEL_CONF = 3
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
    num_classes = 1000
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

    mean_arr = np.zeros(input_shape)
    for img in img_list:
        mean_arr += img.astype('float32')
    mean_arr /= len(img_list)
    print('mean shape:',mean_arr.shape, 'mean mean:',mean_arr.mean(), 'mean max:',mean_arr.max())
    mean_arr /= 255
    np.save('./mean.npy', mean_arr)


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
            xx_train, xx_val, yy_train, yy_val = train_test_split(x_train, y_train, test_size=0.15, random_state=cur_seed,stratify=y_train)
            xx_val = normal_input(xx_val,mean_arr)
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
            best_model_path = './best_model' + str(cur_seed) + '.h5'
            best_model_paths.append(best_model_path)
            checkpoint = ModelCheckpoint(best_model_path,monitor=monitor,verbose=1,save_best_only=True)
            report = report_nsml(prefix = prefix,seed = cur_seed)
            callbacks = [reduce_lr,early_stop,checkpoint,report]

            train_gen = DataGenerator(xx_train, yy_train,fc_train_batch,seq,num_classes,use_aug=True,mean = mean_arr)
            hist1 = model.fit_generator(train_gen,validation_data= (xx_val,yy_val), workers=8, use_multiprocessing=True
                    ,  epochs=fc_train_epoch,  callbacks=callbacks,   verbose=1, shuffle=True)

            for layer in model.layers:
                layer.trainable=True
            model.compile(loss='categorical_crossentropy',  optimizer=opt,  metrics=['accuracy']) 
        
            model.load_weights(best_model_path)
            print('load model:' ,best_model_path)

            train_gen = DataGenerator(xx_train, yy_train,batch_size,seq,num_classes,use_aug=True,mean = mean_arr)
            hist2 = model.fit_generator(train_gen ,validation_data= (xx_val,yy_val), workers=8, use_multiprocessing=True
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
            #why.. low score 0.013....2cv