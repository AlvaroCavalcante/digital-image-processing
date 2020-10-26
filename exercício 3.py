import numpy as np
import matplotlib.pyplot as plt
import os
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

def apply_median_kernel(kernel_area):
    flatten_kernel = kernel_area.flatten()
    sorted_kernel = sorted(flatten_kernel)
    median_index = len(flatten_kernel) // 2
    return sorted_kernel[median_index]

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

kernel_sharpen = np.array([[0, -1, 0], [-1,5,-1], [0,-1,0]]) 
kernel_outline = np.array([[-1, -1, -1], [-1,8,-1], [-1,-1,-1]])
mean = np.full((5,5), 0.11111111111)

median = np.zeros((5,5))

conv_image = convolution(image, median, 1, apply_median_kernel)

noise = np.random.normal(loc=50, scale=30, size=image.shape)

noise_img = image + noise

plt.imshow(noise_img, cmap='gray')

plt.imshow(conv_image, cmap='gray')

row,col = image.shape
s_vs_p = 0.5
amount = 0.1
out = np.copy(image)

num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
out[coords] = 255

num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
out[coords] = 0


gaussian = np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24, 36, 24, 6], [4,16,24,16,4], [1,4,6,4,1]])
