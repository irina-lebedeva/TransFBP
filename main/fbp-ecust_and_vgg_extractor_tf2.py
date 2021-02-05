
import math
import os
import sys
import time

import glob
import random
from xlrd import open_workbook
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import dask.dataframe as dd
import dask.array as da
from util.calc_util import split_train_and_test_data
from util.cfg import config
from util.file_util import mkdirs_if_not_exist, out_result, prepare_scutfbp5500
from util.vgg_face_feature import extract_feature

from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
vgg_model = VGGFace() # pooling: None, avg or max

def extract_feature_tf2(file, layer_name):
   
    out = vgg_model.get_layer(layer_name).output
    vgg_model_new = Model(vgg_model.input, out)
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) # or version=2
    features = vgg_model_new.predict(x)
    return features.ravel()


def ecust_train_and_test_set():
    """
    ECUST-FBP dataset creation
    """
    
    DATA_DIR = '/home/ubuntu/mtcnn_aligned_images/female/'
    LABELS_FILE = '/home/ubuntu/ECUST_FBP/scores/public_generic_all.xlsx'
    
    
    df = pd.read_excel(LABELS_FILE)  
    labels_dict= {}
    for root, dirs, files in os.walk(DATA_DIR):
     
      for file in files: 
        
        f = os.path.join(root, file)
        df1 = df.loc[df['image']==file]
        if (len(df1.index)==1):
         
           labels_dict[f] = float(df1['mean'])
  
      keys =  list(labels_dict.keys())
      random.shuffle(keys)
    d1 = dict(list(labels_dict.items())[len(labels_dict)//2:])
    d2 = dict(list(labels_dict.items())[:len(labels_dict)//2])
    return d1, d2      

def train(train,test)

    train_vec = list()
    train_label = list()
    test_vec = list()
    test_label = list()

    for k, v in train.items():
        feature = np.concatenate((extract_feature_tf2(k, layer_name="conv5_2"), extract_feature_tf2(k,name="conv5_3")),
                                 axis=0)
      
        train_vec.append(feature)
        train_label.append(v)

    for k, v in test.items():
        feature = np.concatenate((extract_feature_tf2(k, layer_name="conv5_2"), extract_feature_tf2(k,name="conv5_3")),
                                 axis=0)
        test_vec.append(feature)
        test_label.append(v)
 
    pca = PCA(n_components=600)
    train_vec = pca.fit_transform(train_vec)
    test_vec = pca.fit_transform(test_vec)    
    reg = linear_model.BayesianRidge()
    #print(train_label)
    train_vec = np.array(train_vec)
    train_label = np.array(train_label)
    reg.fit(train_vec, train_label)
   #mkdirs_if_not_exist('./model')
   #joblib.dump(reg, config['eccv_fbp_reg_model'])

    predicted_label = reg.predict(np.array(test_vec))
    mae_lr = round(mean_absolute_error(np.array(test_label), predicted_label), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(np.array(test_label), predicted_label)), 4)
    pc = round(np.corrcoef(test_label, predicted_label)[0, 1], 4)

    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    csv_tag = time.time()

    mkdirs_if_not_exist('./result')
    df = pd.DataFrame([mae_lr, rmse_lr, pc])
    df.to_csv('./result/performance_%s.csv' % csv_tag, index=False)

    out_result(list(test.keys()), predicted_label.flatten().tolist(), test_label, None,
               path='./result/detail_%s.csv' % csv_tag)



if __name__ == '__main__':


