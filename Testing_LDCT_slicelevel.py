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
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB5
import sys
from numpy import argmax
import pandas as pd
import pdb
import math
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.applications import VGG16, VGG19, ResNet101, DenseNet201, mobilenet, ResNet50, Xception
from tensorflow.keras.applications import EfficientNetB5, EfficientNetB1, EfficientNetB0,InceptionV3,DenseNet121, InceptionResNetV2
import re 
import pandas as pd
import csv

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
        label = 'COVID-19'
    else:
        
        if covid + cap >= 0.75 * normal:
            if covid >= cap:
                label = 'COVID-19'
            else:
                label = 'COVID-19' 
        else:
            label = 'NORMAL'
        
        
    print(name,"  Prediction class = ",label,"  Covid = ",covid," cap=",cap,"  normal",normal)
    return label

         
# In[2]:
model_name = 'ResNet50'
test_folder='All_test'

#Load the path of folder that contains test patient
INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/JournalDraftBaseline/ct_covid19_cap_cnn-main/Preprocessed_Datasets/LDCT'
#'/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_LDCT/'  
patients_test1 = os.listdir(INPUT_FOLDER_test1)


#Loading Model for prediction
def model_dense(image_shape=(512,512,3)):
    

    modelPath = '/cbr/anand/ResearchWork/Covid_2021/JW/ct_covid19_cap_cnn-main/Src_IITM/ModelHistory_New/EfficientNetB5/saved-model-05-1.27.hdf5'#"ModelHistory/Densenet_EfficientNet/saved-model-04-0.22.hdf5"#
    #"ModelHistory/Densenet_EfficientNet/saved-model-04-0.22.hdf5"
    #"ModelHistory/MobileNet_Resnet50/saved-model-05-1.20.hdf5" #
    in_src = Input(shape=image_shape)
    
    m = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(m)
    
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
filename = 'Dataset/LDCT/LDCT_slicelevel_labels_S1.csv'
GT_label_dict ={}
with open(filename,'r') as data:
    for line in csv.reader(data):
        GT_label_dict[line[0]] = line[1:-1]

prediction = []
ground_truth = []
predIdxs_probs = []
patientwise_posterior_probability = {}
patientwise_prediction_label = {}
Final_ground_truth = []
Final_prediction = []
#pdb.set_trace()
for i in tqdm(range(0,len(patients_test1))):
    
    patient = patients_test1[i]
    covid = 0
    cap = 0
    normal = 0
    slicewise_postrior_probability =[]
    patient_slices = os.listdir(os.path.join(INPUT_FOLDER_test1, patient))
    

    start = 0
    end = len(patient_slices)

    patient_name = str(patient)
    
    patient_labels = GT_label_dict[patient_name]
    prediction = []
    ground_truth = []
    slicewise_posterior_probability = []
    for j in range(start,end):
        GT_slice_index = int(((patient_slices[j].split('_'))[1].split('.'))[0])
        
        img = cv2.imread(os.path.join(INPUT_FOLDER_test1, patient, patient_slices[j]))
        img = np.reshape(img,(1,512,512,3))
        predIdxs = model(img)
        slicewise_posterior_probability.append(predIdxs.numpy()[0].reshape(3,1))
        pred = argmax(predIdxs)
        prediction.append(pred)
        val = 0
        if int(patient_labels[GT_slice_index]) == 1:
            val = 2
        ground_truth.append(val)
        predIdxs_probs.append(predIdxs)
        #pdb.set_trace()
    Final_prediction.extend(prediction)
    Final_ground_truth.extend(ground_truth)
    patientwise_posterior_probability[patient_name] = slicewise_posterior_probability
    patientwise_prediction_label[patient_name] = prediction

    



pred  = [1 if x==2 else x for x in Final_prediction]
gt = [1 if x==2 else x for x in Final_ground_truth]
cm= confusion_matrix(gt, pred)
print(cm)

x = ['NORMAL', 'COVID-19']

print(classification_report(gt, pred,target_names=x))
pdb.set_trace()
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(gt_label.argmax(axis=1), pred_label.argmax(axis=1))

if not os.path.exists('Results/LDCT/'):
    os.makedirs('Results/LDCT/')
plt.show()
plt.savefig('Results/LDCT/confusionmatrix_'+ 'LDCT'+'.png')

skplt.metrics.plot_roc_curve(gt_label.argmax(axis=1), pred_patients)
plt.show()
plt.savefig('Results/LDCT/roc_curve'+ '.png')

pdb.set_trace()
