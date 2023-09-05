from lungmask import mask
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pydicom
import os
import pdb
import numpy as np
import cv2
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import math
from scipy.io import savemat
from utils import *
import sys


MAX = 65535
IMG_PX_SIZE = 512

def perform_maskUnetandResizeslices(patients_path, category, patient):

	final_path = patients_path + '/' +  patient
	files = os.listdir(final_path)
	slices = load_scan(final_path)
	image = get_pixels_hu(slices)
	new_img = cropped_slices(image)
	
	final_segmentation = {}
	final_hyperpolarization = {}
	new_img_new = {}

	counter = 0
	for f in files:
		
		input_image = sitk.ReadImage(final_path + '/' + f)
		segmentation = mask.apply(input_image)
		dcmimg = pydicom.dcmread(final_path + '/' + f)
		new_img_new[counter] = (slices[counter].pixel_array)
		final_segmentation[dcmimg.InstanceNumber - 1] = segmentation[0]
		
		counter = counter + 1
		
	mask_ = np.stack([final_segmentation[i] for i in range(0,len(final_segmentation))])
	temp_mask = cropped_slices(mask_)
	new_img_array = np.stack([new_img_new[i] for i in range(0,len(new_img_new))])
	new_img_array = np.array(new_img_array, dtype=np.int16)
	

	new_mask = np.array(temp_mask,dtype=np.int16)
	#new_mask[new_mask[:,:,:]==2] = 1
	
	saveDir = 'TempDir/Matlab_Mask_files_Train_SPGC/' + category + '/' + str(patient)
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	dest_file = saveDir + '/mask.mat'
	
	savemat(dest_file, {"data":new_mask})

	saveDir = 'TempDir/Matlab_Mask_files_Train_SPGC/' + category + '/' +  str(patient)
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	dest_file = saveDir + '/image.mat'
	
	savemat(dest_file, {"data":new_img_array})
	
path = sys.argv[1]


categories = os.listdir(path)
for category in (categories):
	subsets = os.listdir(os.path.join(path, category))

	for subset in subsets:
		patients = os.listdir(os.path.join(path, category, subset))

		for patient in tqdm(patients):

			finalpath = os.path.join(path, category, subset)
			perform_maskUnetandResizeslices(finalpath, category,  patient)

















