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


fast_fourrier_image = np.fft.fft2(image)
fshift = np.fft.fftshift(fast_fourrier_image)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = image.shape
crow,ccol = int(rows/2) , int(cols/2) # ponto de origem 
fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
# fshift[crow+30:rows, ccol+30:cols] = 1
# fshift[crow-30:rows, ccol-30:cols] = 1

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

## aplicando passa baixa
rows, cols = image.shape[0:2]
crow,ccol = rows//2 , cols//2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = fshift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back) #cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

import cv2
# simple averaging filter without scaling parameter
mean_filter = np.ones((3,3))

# creating a guassian filter
x = cv2.getGaussianKernel(5,10)
gaussian = x*x.T

# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# laplacian
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()