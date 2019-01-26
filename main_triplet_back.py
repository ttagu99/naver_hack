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
from keras import Model, Input
from keras.layers import Layer

from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa
import random
from keras.utils.training_utils import multi_gpu_model

class TripletLossLayer(Layer):
	def __init__(self, alpha, **kwargs):
		self.alpha = alpha
		super(TripletLossLayer, self).__init__(**kwargs)

	def triplet_loss(self, inputs):
		a, p, n = inputs
		p_dist = K.sum(K.square(a - p), axis=-1)
		n_dist = K.sum(K.square(a - n), axis=-1)
		return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

	def call(self, inputs):
		loss = self.triplet_loss(inputs)
		self.add_loss(loss)
		return loss

def triplet_loss(y_true,y_pred):
	a,p,n=y_pred[0],y_pred[1],y_pred[2]
	p_dist=K.sum(K.square(a-p),axis=-1)
	n_dist=K.sum(K.square(a-n),axis=-1)
	base_dist=p_dist-n_dist+alpha
	loss=K.sum(K.maximum(base_dist,0))
	return loss

def build_triple_base_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet'):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def build_triple_model(backbone= None, input_shape =  (224,224,3), use_imagenet = 'imagenet', num_classes=1383, opt = SGD()):
    bs_model=build_triple_base_model(backbone= backbone, input_shape =  (224,224,3), use_imagenet =use_imagenet)
    input_a=Input(shape=input_shape, name ='input_anchor')
    input_p=Input(shape=input_shape, name ='input_pos')
    input_n=Input(shape=input_shape, name ='input_neg')
    embedding_a=bs_model(input_a,name='bs_model_anchor')
    embedding_p=bs_model(input_p,name='bs_model_pos')
    embedding_n=bs_model(input_n,name='bs_model_neg')

    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([embedding_a, embedding_p, embedding_n])
    model=Model([input_a,input_p,input_n],triplet_loss_layer)
    model.compile(optimizer=opt,loss=None)
    return model



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
        print('model saved!', os.path.join(dir_name, 'model'))

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!', file_path)

    def infer(queries, db):

        # Query 개수: 195 ->9,014->18,027
        # Reference(DB) 개수: 1,127->36,748
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        # debuging all set size
        # queries = query_img*901
        # references = references*3674
        # query_img = query_img*901
        # reference_img = reference_img*3674

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        #queries = np.asarray(queries)
        #query_img = np.asarray(query_img)
        #references = np.asarray(references)
        #reference_img = np.asarray(reference_img)
        #mean_arr = np.load('./mean.npy')
        query_img = normal_inputs(query_img)
        reference_img = normal_inputs(reference_img)



        infer_img_batch = 100
        TTA = 2
        model.summary()
        intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-2].output)
        print('inference start')

        # inference
        query_veclist=[]
        query_veclist = predict_tta_batch(intermediate_layer_model,query_img,tta=TTA,batch_size=infer_img_batch,use_avg=False)
        query_vecs = np.array(query_veclist)
        print('query_vecs shape:',query_vecs.shape)
        reference_veclist=[]
        if isinstance(reference_img, (list,)):
            print('reference',len(reference_img))
        else:
            print('reference',reference_img.shape)
        reference_veclist = predict_tta_batch(intermediate_layer_model,reference_img,tta=TTA,batch_size=infer_img_batch,use_avg=False)
        reference_vecs = np.array(reference_veclist)
        print('reference_vecs shape:',reference_vecs.shape)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        print(query_vecs.shape, reference_vecs.shape)
        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}
        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            ranked_list = [references[k].split('/')[-1].split('.')[0] for k in indices[i]]
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

def read_image(img_path,shape=(224,224)):
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
        self.set_label_imgs()
        self.on_epoch_end()
        self.mean = mean
        self.img_size = (input_shape[0],input_shape[1])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def get_triple_set(self,index, label):
        anchor_path = self.image_paths[index]
        anchor = read_image(anchor_path, self.img_size)

        if len(self.imgs_per_label[label]) == 1:
            positive_path = self.image_paths[index]
        else:
            while 1:
                ridx = np.random.randint(0,len(self.imgs_per_label[label]))
                positive_path = self.imgs_per_label[label][ridx]
                if positive_path !=anchor_path:
                    break

        positive = read_image(positive_path, self.img_size)


        ## negative sel
        while 1:
            nega_label = np.random.randint(0,self.num_classes)
            
            if nega_label !=label and len(self.imgs_per_label[nega_label]) >=1:
                break

        nega_img_idx = np.random.randint(0,len(self.imgs_per_label[nega_label]))

        ## debuging
        #print(nega_label, nega_img_idx, len(self.imgs_per_label[nega_label]))
        negative_path = self.imgs_per_label[nega_label][nega_img_idx]
        negative =  read_image(negative_path, self.img_size)

        ## debuging
        #print(anchor_path, positive_path, negative_path)

        return anchor, positive, negative 


    def __getitem__(self, index):
        'Generate one batch of data'
        anchors_np=np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        poss_np=np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        negas_np=np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        
        indexes = self.indexes[index*self.batch_size:(index+1)* self.batch_size]

        ancs=[]
        poss=[]
        negs=[]
        for idx in indexes:
            anc,pos,neg = self.get_triple_set(idx,self.labels[idx])
            ancs.append(anc)
            poss.append(pos)
            negs.append(neg)
 
        anchorst = np.array(ancs)
        posst = np.array(poss)
        negast =  np.array(negs)

        ## debugging
        print(anchorst.shape, posst.shape, negast.shape)

        if self.use_aug == True:
            anchors_np[:,:,:,:]  = self.aug_seq.augment_images(anchorst)
            poss_np[:,:,:,:]  = self.aug_seq.augment_images(posst)
            negas_np[:,:,:,:]  = self.aug_seq.augment_images(negast)
        else:
            anchors_np[:,:,:,:]  = anchorst
            poss_np[:,:,:,:]  = posst
            negas_np[:,:,:,:]  = negast

        anchors_np = normal_inputs(anchors_np,self.mean)
        poss_np = normal_inputs(poss_np,self.mean)
        negas_np = normal_inputs(negas_np,self.mean)

        return [anchors_np, poss_np, negas_np], None

    def set_label_imgs(self):
        self.imgs_per_label = {}
        for i in range(self.num_classes):
            self.imgs_per_label.setdefault(i,[])

        for idx, path in enumerate(self.image_paths):
            self.imgs_per_label[self.labels[idx]].append(path)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        org_idx = np.arange(len(self.image_paths))
        # after batch divide, last batch number add
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
        nsml.report(summary=True, epoch=epoch, loss=logs.get('loss'), val_loss=logs.get('val_loss'))
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
    CV_NUM = 1

    CONF_LIST = []
    CONF_LIST.append({'name':'Xc', 'input_shape':(224, 224, 3), 'backbone':Xception
                    , 'batch_size':100 ,'fc_train_batch':330, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'Re50', 'input_shape':(224, 224, 3), 'backbone':ResNet50
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':111,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
    CONF_LIST.append({'name':'TTM', 'input_shape':(224, 224, 3), 'backbone':DenseNet121
                    , 'batch_size':30 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':1, 'fc_train_epoch':1, 'imagenet':None})
    CONF_LIST.append({'name':'IR', 'input_shape':(224, 224, 3), 'backbone':InceptionResNetV2
                    , 'batch_size':100 ,'fc_train_batch':260, 'SEED':222,'start_lr':0.0005
                    , 'finetune_layer':70, 'fine_batchmul':15,'epoch':200, 'fc_train_epoch':10, 'imagenet':'imagenet'})
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

    model = build_triple_model(backbone= backbone, use_imagenet=use_imagenet,input_shape = input_shape, num_classes=num_classes,opt = opt)
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
        #y_train = keras.utils.to_categorical(labels, num_classes=num_classes)

        print(len(labels), 'train samples')


        opt = keras.optimizers.Adam(lr=start_lr)
        #model = build_triple_model(backbone= backbone, use_imagenet=use_imagenet,input_shape = input_shape,opt = opt)
        #xx_train, xx_val, yy_train, yy_val = train_test_split(x_train, y_train, test_size=0.15, random_state=cur_seed,stratify=y_train)
        xx_train, xx_val, yy_train, yy_val = train_test_split(x_train, labels, test_size=0.15, random_state=SEED,stratify=labels)

        print('shape:',xx_train.shape,'val shape:',xx_val.shape)
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
        monitor = 'val_loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=4,factor=0.2,verbose=1)
        early_stop = EarlyStopping(monitor=monitor, patience=7)
        best_model_path = './best_model' + str(SEED) + '.h5'
        checkpoint = ModelCheckpoint(best_model_path,monitor=monitor,verbose=1,save_best_only=True)
        report = report_nsml(prefix = prefix,seed = SEED)
        callbacks = [reduce_lr,early_stop,checkpoint,report]
               
        train_gen = DataGenerator(xx_train,input_shape, yy_train,batch_size,seq,num_classes,use_aug=True,mean = mean_arr)
        val_gen = DataGenerator(xx_val,input_shape, yy_val,batch_size,seq,num_classes,use_aug=False,shuffle=False, mean = mean_arr)
        hist = model.fit_generator(train_gen ,validation_data= val_gen, workers=1, use_multiprocessing=False
                    ,  epochs=nb_epoch,  callbacks=callbacks,   verbose=1, shuffle=True)

        model.load_weights(best_model_path)
        nsml.report(summary=True)
        nsml.save(prefix +'BST')