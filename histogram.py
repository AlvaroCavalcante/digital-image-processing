from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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


def instantiate_histogram():    
    hist_array= []
    
    for i in range(0,256):
        hist_array.append(str(i))
        hist_array.append(0)
    
    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range(0, len(hist_array), 2)} 
    
    return hist_dct

histogram = instantiate_histogram()

# image = np.matrix(image) / 255

def plot_hist(hist):
    plt.bar(hist.keys(), hist.values())
    
    plt.xlabel("Níveis intensidade")
    
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    
    plt.grid(True)
    plt.show()
    
n_pixels = image.shape[0] * image.shape[1]

def count_intensity_values(hist):
    for row in image:
        for column in row:
            hist[str(column)] = hist[str(column)] + 1
     
    return hist       

histogram = count_intensity_values(histogram)
plot_hist(histogram)

def test_hist_sum(hist, image_pixels):
    total = 0
    for i in range(0, 256):
        total += hist[str(i)]

    if total != image_pixels:
        print('Different number of pixels', total, image_pixels)

test_hist_sum(histogram, n_pixels)
    
def get_percentagem_hist(hist, n_pixels):
    percent_hist = {}
    for i in range(0, 256):
        percent_hist[str(i)] = hist[str(i)] / n_pixels
    
    return percent_hist

percent_hist = get_percentagem_hist(histogram, n_pixels)
    
def test_hist_sum_perc(hist_perc):
    total_perc = 0
    for i in range(0, 256):
        total_perc += hist_perc[str(i)]

    if total_perc != 1:
        print('Percentagens sum is not 1:', total_perc)
    
test_hist_sum_perc(percent_hist)

def get_accumulated_perc(percent_hist): 
    acc_perc = {}
    sum_perc = 0
    
    for i in range(0, 256):
        if i == 0:
            pass
        else: 
            sum_perc += percent_hist[str(i - 1)]
            
        acc_perc[str(i)] = percent_hist[str(i)] + sum_perc

    return acc_perc

accumulated_perc = get_accumulated_perc(percent_hist)

def get_new_gray_value(acc_perc):
    new_gray_value = {}
    
    for i in range(0, 256):
        new_gray_value[str(i)] = np.ceil(acc_perc[str(i)] * 255)

    return new_gray_value

new_gray_value = get_new_gray_value(accumulated_perc)

def equalize_img(img, new_gray_value):
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = new_gray_value[str(img[row][column])]
            
    return img

image = equalize_img(image, new_gray_value)
plt.imshow(image, cmap='gray')

histogram = instantiate_histogram()
histogram = count_intensity_values(histogram)
plot_hist(histogram)
