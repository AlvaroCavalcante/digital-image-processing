import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import data
import cv2 

image = data.brick()

image = data.astronaut()

def convert_to_gray(image, luma=False):
    if luma:
        params = [0.299, 0.589, 0.114]
    else:
        params = [0.2125, 0.7154, 0.0721]

    gray_image = np.ceil(np.dot(image[...,:3], params))
    gray_image[gray_image > 255] = 255
    
    return gray_image.astype('uint8')

image = convert_to_gray(image)
plt.imshow(image, cmap='gray')

def get_img_with_padding(image, kernel):
    hor_stack = 0
    while ((kernel.shape[0] - 1) - hor_stack) != 0: 
        image = np.insert(image, 0, 0, axis=0)
        image = np.vstack([image, np.zeros(image.shape[1])])
        hor_stack += 1
        
    vert_stack = 0
    while ((kernel.shape[0] - 1) - vert_stack) != 0: 
        image = np.insert(image, 0, 0, axis=1)
        image = np.hstack([image, np.zeros((image.shape[0], 1))])
        vert_stack += 1
        
    return image

def convolution(image, kernel, stride, function=False ,padding=False):
    initial_line = 0
    final_line = kernel.shape[0]
    new_image = []

    if padding:
        image = get_img_with_padding(image, kernel)
    
    while final_line <= image.shape[0]:    
        initial_column = 0
        final_column = kernel.shape[1]
        matrix_line = []

        while final_column <= image.shape[1]:
            kernel_area = image[initial_line:final_line, initial_column:final_column]
            
            if function:
                matrix_line.append(function(kernel_area))
            else:                   
                matrix_line.append(np.sum(kernel * kernel_area))
        
            initial_column += stride 
            final_column += stride
        
        new_image.append(matrix_line) 
        final_line += 1
        initial_line += 1

    return np.asmatrix(new_image)

# sobel in x direction
sobel_y= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

# sobel in y direction
sobel_x= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])

prewitt_x = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])

prewitt_y = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])

roberts_1 = np.array([[1,0], [0,-1]])

roberts_2 = np.array([[0,-1], [1,0]])


laplacian = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])

laplacian2 = np.array([[-1,-4,-1], [-4,20,-4], [-1,-4,-1]])

mean = np.full((5,5), 0.11111111111)
cv1 = convolution(image, mean, 1)
conv_image = convolution(image, sobel_x, 1)
plt.imshow(conv_image, cmap='gray')

laplaciano_gau = np.array([[0,0,-1,0,0], [0,-1,-2,-1,0], [-1,-2,16,-2,-1], [0,-1,-2,-1,0], [0,0,-1,0,0]])

conv_image = convolution(image, laplaciano_gau, 1)
plt.imshow(conv_image, cmap='gray')


def binarize_img(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 1
    return img

bin_image = binarize_img(image.copy(), 150)
plt.imshow(bin_image, cmap='gray')

ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(th2, cmap='gray')
