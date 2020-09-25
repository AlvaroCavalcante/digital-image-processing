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

def binarize_img(image, threshold):
    image[image < threshold] = 0
    image[image >= threshold] = 1
    return image

bin_image = binarize_img(image.copy(), 50)
plt.imshow(bin_image, vmin=0, vmax=1, cmap='gray')

