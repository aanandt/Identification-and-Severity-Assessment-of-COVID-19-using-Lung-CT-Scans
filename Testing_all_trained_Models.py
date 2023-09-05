import scipy.io
import numpy as np
import pandas as pd
 
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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB5, EfficientNetB1, EfficientNetB0
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
from tensorflow.keras.applications import ResNet101, mobilenet, ResNet50


IMG_PX_SIZE = 512
def three_channel(img):
    img = np.stack((img,)*3, axis=-1)
    
    return img


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
        label = 'CAP'#'COVID-19'#'CAP'
    else:
        
        if covid + cap >= 0.75 * normal:
            if covid >= cap:
                label = 'COVID-19'
            else:
                label = 'CAP' 
        else:
            label = 'NORMAL'
        
        
    print(name,"  Prediction class = ",label,"  Covid = ",covid," cap=",cap,"  normal",normal)
    return label

         
# In[2]:
model_name = 'ResNet50'
test_folder='All_test'

#Load the path of folder that contains test patient
#INPUT_FOLDER_test1 = 'Preprocessed_Data_Mosmed/'  
#patients_test1 = os.listdir(INPUT_FOLDER_test1)
#for TIFF and other
#patients_test1.sort(key=num_sort)
#for LDCT dataset
#patients_test1.sort(key=num_sort)


#Load the path of folder that contains test patient(SPGC)
INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/JournalDraftBaseline/ct_covid19_cap_cnn-main/Preprocessed_Datasets/SPGC/Test/'
#'/cbr/anand/ResearchWork/Covid_2021/Preprocessed_Datasets/MY_JPG/Test/FInalTest/'  
patients_test1 = os.listdir(INPUT_FOLDER_test1)
patients_test1.sort()
gt_filename = '/cbr/anand/ResearchWork/Covid_2021/Dataset/Test/final_label.csv'
output_filename = 'Test_result_SPGC.csv'

# INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_AbelationStudy/SPGC/GMM_filter_img'  
# patients_test1 = os.listdir(INPUT_FOLDER_test1)
# patients_test1.sort()
# gt_filename = '/cbr/anand/ResearchWork/Covid_2021/Dataset/Test/final_label.csv'
# #'/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Results_LDCT.csv'
# #'/cbr/anand/ResearchWork/Covid_2021/Dataset/Test/final_label.csv'
# #
# output_filename = 'Test_result_Ablation.csv'

#Load the path of folder that contains test patient(LDCT)
# INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_LDCT/'  
# patients_test1 = os.listdir(INPUT_FOLDER_test1)
# patients_test1.sort()
# gt_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Results_LDCT.csv'
# output_filename = 'Test_result_LDCT.csv'

#Load the path of folder that contains test patient(LDCT PCR)
# INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_LDCT_PCR/'  
# patients_test1 = os.listdir(INPUT_FOLDER_test1)
# patients_test1.sort()
# gt_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Results_LDCT_PCR_New.csv'
# output_filename = 'Test_result_LDCT_PCR.csv'

#Load the path of folder that contains test patient(TIFF)
# INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_TIFF_New/'  
# patients_test1 = os.listdir(INPUT_FOLDER_test1)
# patients_test1.sort(key=num_sort)
# gt_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Result_TIFFData.csv'
# output_filename = 'Test_result_Mosemed.csv'

#Load the path of folder that contains test patient(STOIC)
# INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_Mosmed/'  
# patients_test1 = os.listdir(INPUT_FOLDER_test1)
# patients_test1.sort(key=num_sort)
# gt_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Results_Mosmed.csv'
# output_filename = 'Test_result_Mosmed.csv'


# #Load the path of folder that contains test patient(Mehta)
# INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_Mehta/'  
# patients_test1 = os.listdir(INPUT_FOLDER_test1)
# patients_test1.sort()
# gt_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Results_Mehta.csv'
# output_filename = 'Test_result_Mehta.csv'

#Load the path of folder that contains test patient(TCIA)
# INPUT_FOLDER_test1 = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Preprocessed_Data_TCIA/'  
# patients_test1 = os.listdir(INPUT_FOLDER_test1)
# patients_test1.sort()
# gt_filename = '/cbr/anand/ResearchWork/Covid_2021/JW/My_Model/Mehta/Results_TCIA.csv'
# output_filename = 'Test_result_TCIA.csv'

#pdb.set_trace()
#Loading Model for prediction
def model_dense(image_shape=(IMG_PX_SIZE,IMG_PX_SIZE,3)):

   #path =  'saved-model-05-0.37.hdf5'
    modelPath = '/cbr/anand/ResearchWork/Covid_2021/JW/ct_covid19_cap_cnn-main/Src_IITM/ModelHistory_New/EfficientNetB5/saved-model-05-1.27.hdf5'
                #'/cbr/anand/ResearchWork/Covid_2021/JW/ct_covid19_cap_cnn-main/Src/saved-model-05-0.35.hdf5'
                #"ModelHistory/Densenet_EfficientNet/saved-model-04-0.22.hdf5" 
                #'ModelHistory/Densenet_EfficientNet/saved-model-04-0.22.hdf5'
                #'/cbr/anand/ResearchWork/Covid_2021/JW/ct_covid19_cap_cnn-main/Src_IITM/ModelHistory_New/ResNet101/saved-model-05-0.88.hdf5'
                #'/cbr/anand/ResearchWork/Covid_2021/JW/ct_covid19_cap_cnn-main/Src_IITM_AblationStudy_Updated/GMM_imfill_NoBoundary/Model_History_AblationStudy/GMM_filtered_img/' + path
                #'/cbr/anand/ResearchWork/Covid_2021/JW/ct_covid19_cap_cnn-main/Src_IITM/ModelHistory_New/ResNet101/saved-model-05-0.88.hdf5'
                #"ModelHistory/Densenet_EfficientNet/saved-model-04-0.22.hdf5"
                #'/cbr/anand/ResearchWork/Covid_2021/JW/ct_covid19_cap_cnn-main/Src_IITM/ModelHistory_New/ResNet50/saved-model-04-0.22.hdf5'
                #"ModelHistory/Densenet_EfficientNet/saved-model-04-0.22.hdf5"
                #"ModelHistory/MobileNet_Resnet50/saved-model-05-1.20.hdf5" #
    in_src = Input(shape=image_shape)
    
    m = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(IMG_PX_SIZE,IMG_PX_SIZE,3))(in_src)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(m)
    
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    
    x = Dense(3, activation='softmax')(x)
    model = Model(in_src,x)
    model.load_weights(modelPath)#("saved-model-03-0.51.hdf5")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
model = model_dense()


#pdb.set_trace()
pred_dict = {}
data = []
#pdb.set_trace()
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
        img_1 = cv2.resize(np.array(img[:,:,0]), (IMG_PX_SIZE,IMG_PX_SIZE))
        img = three_channel(img_1)
        img = np.reshape(img,(1,IMG_PX_SIZE,IMG_PX_SIZE,3))
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
import seaborn as sns

pdb.set_trace()

df = pd.DataFrame(data, columns = ['PatientName', 'Labels'])
df.to_csv(output_filename)

df = pd.read_csv(gt_filename)
gt= list(df['Labels'])
pred = list(pred_dict.values())
cm= confusion_matrix(gt, pred)

cm = confusion_matrix(gt, pred)

ax = plt.subplot()
sns.set(font_scale=3.0) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g", cbar=False);  

# Labels, title and ticks
label_font = {'size':'14'}  # Adjust to fit
ax.set_xlabel('Predicted labels', fontdict=label_font);
ax.set_ylabel('True labels', fontdict=label_font);

title_font = {'size':'14'}  # Adjust to fit
ax.set_title('Confusion Matrix', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust to fit
ax.xaxis.set_ticklabels(['COVID-19' , 'Normal']);
ax.yaxis.set_ticklabels([ 'COVID-19', 'Normal']);
plt.show()
plt.savefig('CM_CT3_Mosmed.png')
plt.close()
print(cm)
x = ['COVID-19', 'CAP', 'NORMAL']#['COVID-19', 'NORMAL']#['COVID-19', 'NORMAL']#['COVID-19', 'CAP', 'NORMAL']#
lb = LabelBinarizer()
lb.fit(x)


gt_label = lb.transform(gt)
pred_label = lb.transform(pred)
print(classification_report(gt_label, pred_label,target_names=lb.classes_))
pdb.set_trace()
import scikitplot as skplt
#skplt.metrics.plot_confusion_matrix(gt_label.argmax(axis=1), pred_label.argmax(axis=1))
#categories = ['COVID', 'Normal', 'CAP']
if not os.path.exists('Results/My_Model_results'):
    os.makedirs('Results/My_Model_results/')
#skplt.metrics.plot_confusion_matrix(gt_label.argmax(axis=1), pred_label.argmax(axis=1))
skplt.metrics.plot_confusion_matrix(gt_label, pred_label)
plt.show()
plt.savefig('Results/My_Model_results/ResNet101_confusionmatrix'+'_Mosmed'+ '.png')
pdb.set_trace()
skplt.metrics.plot_roc_curve(gt_label, pred_label)
plt.show()
plt.savefig('Results/My_Model_results/EfficientNetB5_ROC_curve_LDCT'+ '.png')


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
