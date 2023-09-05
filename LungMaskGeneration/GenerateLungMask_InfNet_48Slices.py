from lungmask import mask_Mosmed
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
import nibabel as nib
from utils import *
import sys

MAX = 65535
IMG_PX_SIZE = 512

def perform_maskUnetandResizeslices(patients_path, patient):

	final_path = patients_path + '/' +  patient
	input_image = sitk.ReadImage(final_path)
	inimg_raw = sitk.GetArrayFromImage(input_image)
	inimg_raw[inimg_raw < -1024] = -1024

	counter = 1
	inimg_raw = cropped_slices(inimg_raw)

	new_img={}
	final_segmentation = {}

	for image in tqdm(inimg_raw):

		segmentation = mask_Mosmed.apply(inimg_raw[counter -1, :, :], np.asarray(input_image[counter - 1, : ,:].GetDirection()))
		#segmentation = mask_Mosmed.apply_fused(inimg_raw[counter -1, :, :], np.asarray(input_image[counter - 1, : ,:].GetDirection()))
		final_segmentation[counter] = segmentation[0]
		
		new_img[counter] = inimg_raw[counter - 1, :, :]
		counter = counter + 1

	array = np.stack([final_segmentation[i] for i in range(1,len(final_segmentation)+1)])
	temp_mask = cropped_slices(array)
	new_mask = np.array(temp_mask,dtype=np.int16)
	new_mask[new_mask[:,:,:]==2] = 1

	org_image = np.stack([new_img[i] for i in range(1,len(new_img)+1)])
	temp_img = cropped_slices(org_image)
	#pdb.set_trace()
	new_image = np.array(temp_img,dtype=np.int16)

	saveDir = 'TempDir/Matlab_Mask_files_Test_InfiNetData_638Slices/' + str(patient)
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	dest_file = saveDir + '/mask.mat'
	savemat(dest_file, {"data":new_mask})

	saveDir = 'TempDir/Matlab_Mask_files_Test_InfiNetData_638Slices/' + str(patient)
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	dest_file = saveDir + '/image.mat'
	savemat(dest_file, {"data":new_image})
	#pdb.set_trace()
	
path = sys.argv[1]
# path = '../../Dataset/InfNet/Test/' ## 100 slices
patient = 'tr_im.nii.gz'
perform_maskUnetandResizeslices(path, patient)
			
