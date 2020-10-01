import numpy as np
import matplotlib.pyplot as plt 
from skimage import data
# from PIL import Image
import math 

image = data.coffee()
image2 = data.brick()
image3 = data.brick()

def convert_to_gray(image, luma=False):
    # http://poynton.ca/PDFs/ColorFAQ.pdf
    image_r = image[:,:,0]
    image_g = image[:,:,1]
    image_b = image[:,:,2]
 
    if luma:
        params = [0.299, 0.589, 0.114]
    else:
        params = [0.2125, 0.7154, 0.0721]

    gray_image = np.ceil(np.dot(image[...,:3], params))
    gray_image[gray_image > 255] = 255
    
    return gray_image

# x = np.ceil((image_r + image_g + image_b) / 3)

image = convert_to_gray(image)
plt.imshow(image, cmap='gray')

def generate_gradient(image, horizontal=True):
    pixel_progress = image.shape[0] // 256
    
    count = 0 
    pixel_value = 0 
    
    for i in range(image.shape[0]):
        if (count / pixel_progress).is_integer():
            pass
        else:
            pixel_value += 1
            
        if horizontal:
            image[:,count] = pixel_value
        else:
            image[count] = pixel_value
        count += 1    
    
    return image

def normalize_image(image):
    image[image > 255] = 255
    image[image < 0] = 0
    return image


def sum_matrix(*args):    
    count = 0
    
    for i in range(len(args) - 1):
        if count == 0:
            summed_image = (args[count] + args[count + 1])
        else:
            summed_image += args[count + 1]
        count += 1
       
    norm_img = np.ceil(summed_image / len(args))         
    
    return normalize_image(norm_img)
    
horiz_gradient = generate_gradient(image2)
plt.imshow(horiz_gradient, cmap='gray')
           
new_image = sum_matrix(image, horiz_gradient)
plt.imshow(new_image, cmap='gray')

vert_gradient = generate_gradient(image3, False)      
plt.imshow(vert_gradient, cmap='gray')

new_image2 = sum_matrix(image, vert_gradient)
plt.imshow(new_image2, cmap='gray')

combined_image = sum_matrix(image, horiz_gradient, vert_gradient) 
plt.imshow(combined_image, cmap='gray')


diff_degrade = horiz_gradient - vert_gradient
plt.imshow(diff_degrade, cmap='gray')



def rotate_image(image, radius=0.2):
    new_image = np.zeros((image.shape[0], image.shape[1]))
   
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):   
            x = row * math.cos(radius) - column * math.sin(radius)
            y = row * math.sin(radius) + column * math.cos(radius) 
            
            try:
                if int(x) < 0 or int(y) < 0:
                    raise Exception
                new_image[row][column] = image[int(x)][int(y)]
            except:
                new_image[row][column] = 0

    return new_image

rotated_img = rotate_image(image)
plt.imshow(rotated_img, cmap='gray')

def scale_image(image, scale_factor=0.5):
    new_image = np.zeros((image.shape[0], image.shape[1]))
   
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):   
            x = row * scale_factor
            y = column * scale_factor
            
            try:
                if int(x) < 0 or int(y) < 0:
                    raise Exception
                new_image[row][column] = image[int(x)][int(y)]
            except:
                new_image[row][column] = 0

    return new_image


rotated_img = scale_image(image, 0.5)
plt.imshow(rotated_img, cmap='gray')


def translate_image(image, dx=52.5, dy=32.3):
    new_image = np.zeros((image.shape[0], image.shape[1]))
   
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):   
            x = row - dx
            y = column - dy
            
            try:
                if int(x) < 0 or int(y) < 0:
                    raise Exception
                new_image[row][column] = image[int(x)][int(y)]
            except:
                new_image[row][column] = 0

    return new_image


translated_img = translate_image(image, 50, 100)

translated_img2 = translate_image(image, 50, 100)

from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
fig = figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(2,2,1)
plt.imshow(translated_img, cmap='gray')
ax2 = fig.add_subplot(2,2,2)
plt.imshow(translated_img2, cmap='gray')

        
def shear_image(image, shear_v=0.3, shear_h=0):
    new_image = np.zeros((image.shape[0], image.shape[1]))
   
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):   
            x =  row + (column * shear_v)
            y = (row * shear_h) + column
            try:
                if int(x) < 0 or int(y) < 0:
                    raise Exception
                new_image[row][column] = image[int(x)][int(y)]
            except:
                new_image[row][column] = 0

    return new_image


shear_img = shear_image(image)
plt.imshow(shear_img, cmap='gray')
