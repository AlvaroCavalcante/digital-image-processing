import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os
from keras.preprocessing import image
from skimage import data

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

def convolution(image, kernel, stride, padding = False):
    initial_line = 0
    final_line = kernel.shape[0]
    new_image = []

    if padding:
        image = get_img_with_padding(image) # segundo livro preenchimento abaixo= m - 1 e coluna esquerda e direita de n - 1
    
    while final_line <= image.shape[0]:    
        initial_column = 0
        final_column = kernel.shape[1]
        matrix_line = []

        while final_column <= image.shape[1]:
            kernel_area = image[initial_line:final_line, initial_column:final_column]
                                    
            matrix_line.append(np.sum(kernel * kernel_area))
        
            initial_column += stride 
            final_column += stride
        
        new_image.append(matrix_line) 
        final_line += 1
        initial_line += 1

    return np.asmatrix(new_image)

kernel_sharpen = np.array([[0, -1, 0], [-1,5,-1], [0,-1,0]]) 
kernel_outline = np.array([[-1, -1, -1], [-1,8,-1], [-1,-1,-1]])

# image = image.load_img(path + filepath, target_size=size)

conv_image = convolution(image, kernel_outline, 1)

plt.imshow(conv_image, cmap='gray')
