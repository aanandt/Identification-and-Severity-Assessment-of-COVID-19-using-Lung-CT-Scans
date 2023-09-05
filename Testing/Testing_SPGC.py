

import scipy.io
import numpy as np
import pandas as pd
import pydicom 
from numpy import savez_compressed
import os
import scipy.ndimage
import matplotlib.pyplot as plt

import cv2
import mpl_toolkits
import re

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from numpy.random import randint
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

from matplotlib import pyplot
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from numpy import load
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5,EfficientNetB1
import sys

from numpy import argmax
import pandas as pd
import pdb
import math
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.applications import ResNet50, ResNet101


def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

MAX = 65535

def weighted_majority_vote(pred,name):
    covid=0
    normal=0
    cap=0
    
    onethird = int(len(pred)/3)
    twothird = onethird + onethird
    for i in pred[0:onethird]:
        if(i==2):
            covid+=0.7
        elif(i==1):
            cap+=0.7
        else:
            normal+=0.5
    for i in pred[twothird:]:
        if(i==2):
            covid+=0.7
        elif(i==1):
            cap+=0.7
        else:
            normal+=0.5
    for j in pred[onethird:twothird]:
        if(j==2):
            covid+=1
        elif(j==1):
            cap+=1
        else:
            normal+=1 
    
    if(covid>= normal and covid>=cap):
        label = 'COVID-19'
    elif(cap>covid and cap>normal):
        label = 'CAP'
    else:
        
        if covid + cap >= 0.75 * normal:
            if covid >= cap:
                label = 'COVID-19'
            else:
                label = 'CAP' 
        else:
            label = 'NORMAL'
        
        #label = 'NORMAL'
    print(name,"  Prediction class = ",label,"  Covid = ",covid," cap=",cap,"  normal",normal)
    return label

         

#Load the path of folder that contains test patient
INPUT_FOLDER_test1 =  sys.argv[1]
patients_test1 = os.listdir(INPUT_FOLDER_test1)
patients_test1.sort()
#pdb.set_trace()
model_path = sys.argv[2]
feature_extractor = sys.argv[3]
gt_filename = sys.argv[4]

#Loading Model for prediction
def model_dense(model_path, feature_extractor):

    image_shape=(512,512,3)
    modelPath = model_path
    in_src = Input(shape=image_shape)
    
    if feature_extractor == 'EfficientNetB5':
        m = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    elif feature_extractor == 'EfficientNetB1':
        m = EfficientNetB1(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    elif feature_extractor == 'ResNet50':
        m = ResNet50(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    elif feature_extractor == 'ResNet101':
        m = ResNet101(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    else:
        print("Select the feature extractor")
    
    x = tf.keras.layers.GlobalAveragePooling2D()(m)
 
    x = Dense(2048,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(in_src,x)
    model.load_weights(model_path)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
model = model_dense(model_path, feature_extractor)



pred_dict = {}
data = []

for i in range(0,len(patients_test1)):
    
    patient = patients_test1[i]
    covid = 0
    cap = 0
    normal = 0
    prediction = []
    predIdxs_probs = []
    patient_slices = os.listdir(os.path.join(INPUT_FOLDER_test1, patient))
    
    start = 0
    end = len(patient_slices)

    patient_name = str(patient)

    for j in range(start,end):
        img = cv2.imread(os.path.join(INPUT_FOLDER_test1, patient, patient_slices[j]))
        
        img = np.reshape(img,(1,512,512,3))
        predIdxs = model(img)
        pred = argmax(predIdxs)
        prediction.append(pred)
        predIdxs_probs.append(predIdxs)
        

    label = weighted_majority_vote(prediction,patient_name)
    pred_dict[patient_name] = label
    data.append([patient_name,label])

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

if not os.path.exists('ClassificationResults'):
    os.makedirs('ClassificationResults')
df = pd.DataFrame(data, columns = ['PatientName', 'Labels'])
df.to_csv('ClassificationResults/Test_result_SPGC_'+feature_extractor+'.csv')

df = pd.read_csv(gt_filename)
gt= list(df['Labels'])
pred = list(pred_dict.values())
cm= confusion_matrix(gt, pred)
print(cm)
x = ['COVID-19', 'CAP', 'NORMAL']
lb = LabelBinarizer()
lb.fit(x)


gt_label = lb.transform(gt)
pred_label = lb.transform(pred)
print(classification_report(gt_label, pred_label,target_names=lb.classes_))
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(gt_label.argmax(axis=1), pred_label.argmax(axis=1))

plt.show()
plt.savefig('ClassificationResults/confusionmatrix_'+ 'SPGC_'+feature_extractor+'.png')

skplt.metrics.plot_roc_curve(gt_label.argmax(axis=1), pred_patients)
plt.show()
plt.savefig('ClassificationResults/roc_curve_SPGC_'+feature_extractor+ '.png')

