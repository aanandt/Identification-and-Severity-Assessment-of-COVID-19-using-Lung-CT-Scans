#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import scipy.io
import numpy as np
import pandas as pd
import pydicom 
from numpy import savez_compressed
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import pylibjpeg
import cv2
import mpl_toolkits
import re
#import pylibjpeg
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from numpy.random import randint
from tensorflow.keras.models import Model
#from keras.models import Input
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import np_utils
from matplotlib import pyplot
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from numpy import load
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB5
import sys
import matplotlib.pyplot as plt
from numpy import argmax
import pandas as pd
import pdb
import math
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, DenseNet201, mobilenet, ResNet50
import re 

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

         
# In[2]:
model_name = 'ResNet50'
test_folder='All_test'

#Load the path of folder that contains test patient
INPUT_FOLDER_test1 = 'Preprocessed_Data_Mehta/'  
patients_test1 = os.listdir(INPUT_FOLDER_test1)
pdb.set_trace()
#for TIFF and other
#patients_test1.sort(key=num_sort)
#for LDCT dataset
#patients_test1.sort(key=num_sort)
#for Mehta Dataset
patients_test1.sort()


#Loading Model for prediction
def model_dense(image_shape=(512,512,3)):

    modelPath = "ModelHistory/Densenet_EfficientNet/saved-model-04-0.22.hdf5"#"ModelHistory/MobileNet_Resnet50/saved-model-05-1.20.hdf5" #
    in_src = Input(shape=image_shape)
    #d = BatchNormalization()(in_src)
    m = ResNet50(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    #Here I am making all the layer of the last layer to be non trainable
    # for layer in m.layers[:len(m.layers)-46]:
    #     layer.trainable = False
    #x = tf.keras.layers.GlobalMaxPool2D()(model)
    x = tf.keras.layers.GlobalAveragePooling2D()(m)
    #x = Flatten()(model)
    x = Dense(2048,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(in_src,x)
    model.load_weights(modelPath)#("saved-model-03-0.51.hdf5")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
model = model_dense()



pred_dict = {}
data = []
pdb.set_trace()
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
        #pdb.set_trace()

    label = weighted_majority_vote(prediction,patient_name)
    pred_dict[patient_name] = label
    data.append([patient_name,label])

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
pdb.set_trace()

df = pd.DataFrame(data, columns = ['PatientName', 'Labels'])
df.to_csv('Test_result_Mehta.csv')

df = pd.read_csv('Results_Mehta.csv')
gt= list(df['Labels'])
pred = list(pred_dict.values())
cm= confusion_matrix(gt, pred)
print(cm)
x = ['COVID-19', 'NORMAL']#['COVID-19', 'CAP', 'NORMAL']
lb = LabelBinarizer()

lb.fit(x)

gt_label = lb.transform(gt)
pred_label = lb.transform(pred)
print(classification_report(gt_label, pred_label,target_names=lb.classes_))
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(gt_label.argmax(axis=1), pred_label.argmax(axis=1))
#categories = ['COVID', 'Normal', 'CAP']
if not os.path.exists('Results/IITM'):
    os.makedirs('Results/IITM/')
plt.show()
plt.savefig('Results/IITM/confusionmatrix_'+ 'Mehta'+'.png')

skplt.metrics.plot_roc_curve(gt_label.argmax(axis=1), pred_patients)
plt.show()
plt.savefig('Results/IITM/roc_curve'+ '.png')


'''
GT_patients = []
pred_patients = []
for label in pred:
    if label == 'CAP':
        entry = [1,0,0]   
    if label == 'COVID-19':

        entry = [0,1,0]
    if label == 'NORMAL':
        entry == [0,0,1]
    pred_patients.append(entry)

for label in gt:
    if label == 'CAP':
        entry = [1,0,0]   
    if label == 'COVID-19':

        entry = [0,1,0]
    if label == 'NORMAL':
        entry == [0,0,1]
    GT_patients.append(entry)

pdb.set_trace()
pred_labels =[]
for patient in pred_dict.keys():
    pred_labels.append(pred_dict[patient])
lb = LabelBinarizer()
pred_patients = lb.transform(pred_labels)

print(classification_report(gt_label, pred_patients,target_names=lb.classes_))

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(gt_label.argmax(axis=1), pred_patients.argmax(axis=1))
#categories = ['COVID', 'Normal', 'CAP']
if not os.path.exists('Results/saved-model-03-0.51.hdf5'):
    os.makedirs('Results/saved-model-03-0.51.hdf5/')
plt.show()
plt.savefig('Results/saved-model-03-0.51.hdf5/confusionmatrix_'+ 'LDCT'+'.png')

skplt.metrics.plot_roc_curve(gt_label.argmax(axis=1), pred_patients)
plt.show()
plt.savefig('Results/saved-model-03-0.51.hdf5/roc_curve'+ '.png')

'''
pdb.set_trace()
