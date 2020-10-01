import numpy as np
import matplotlib.pyplot as plt 
from skimage import data
from PIL import Image
import math 

image = data.coffee()

def convert_to_gray(image, luma=False):
    if luma:
        params = [0.299, 0.589, 0.114]
    else:
        params = [0.2125, 0.7154, 0.0721]

    gray_image = np.ceil(np.dot(image[...,:3], params))
    gray_image[gray_image > 255] = 255
    
    return gray_image

image = convert_to_gray(image)
plt.imshow(image, cmap='gray')

def binarize_img(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 1
    return img

bin_image = binarize_img(image.copy(), 50)
plt.imshow(bin_image, vmin=0, vmax=1, cmap='gray')

def exponential_trans(img):
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = np.exp(img[row][column])
            
    return img

exp_image = exponential_trans(image.copy())
plt.imshow(exp_image)

def normalize_img(img):
    img /= np.max(img)
    img = np.ceil(img * 255)  
    return img

def potential_trans(img, gamma):
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = img[row][column] ** gamma


    return normalize_img(img)

pot_image = potential_trans(image.copy(), 0.2)
plt.imshow(pot_image, cmap='gray')

def log_trans(img):
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = math.log(img[row][column], 2)
            
    return img

log_image = log_trans(image.copy())
plt.imshow(log_image)

def sqrt_trans(img):
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = math.sqrt(img[row][column])
            
    return img

sqrt_image = sqrt_trans(image.copy())
plt.imshow(sqrt_image)
