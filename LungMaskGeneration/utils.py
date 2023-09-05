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


IMG_PX_SIZE = 512
MAX = 65535

# Load the scans in given folder path

def load_scan(path):
    
  slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
  slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
  try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
  except:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

  for s in slices:
    s.SliceThickness = slice_thickness
  return slices

def get_pixels_hu(slices):

  image = np.stack([s.pixel_array for s in slices])
  # should be possible as values should always be low enough(<32k)
  image = image.astype(np.int16)

  #Set outside-of-scan pixels to 0
  # The intercept is usually -1024, so air is approximately 0 
  image[image == -2000] = 0

  #Convert to Hounsfield units(HU)
  for slice_number in range(len(slices)):

    intercept = slices[slice_number].RescaleIntercept
    slope = slices[slice_number].RescaleSlope

    if slope != 1:
      image[slice_number] = slope * image[slice_number].astype(np.float64)
      image[slice_number] = image[slice_number].astype(np.int16)

    image[slice_number] += np.int16(intercept)

  return np.array(image, dtype=np.int16)

def three_channel(img):
    #img = cv2.resize(img,(256,256))
    img = np.stack((img,)*3, axis=-1)
    
    return img

def normalize(volume):
    """Normalize the volume"""
    min = -1150
    max = 150
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = (volume*255).astype("uint8")
    return volume

def cropped_slices(slices):

	new_slices = []
	for num,each_slice in enumerate(slices):

		new_img = cv2.resize(np.array(each_slice),(IMG_PX_SIZE,IMG_PX_SIZE))
		new_slices.append(new_img)

	array = np.stack([new_slice for new_slice in new_slices])
	return array

