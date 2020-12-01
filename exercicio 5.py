import cv2 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

img = np.array(Image.open('/home/alvaro/Downloads/original/bat-8.gif'))

plt.imshow(img, cmap='gray')

img = np.where(img==255, 1, img) 

def map_coordinates(img):
    coords = []
    for col in range(img.shape[0]):
        for line in range(img.shape[1]):    
            if img[col][line] == 1:
                coords.append((col, line)) 
    
    return coords

def dilate_img(coords, img):
    dilated_img = img.copy()
    try:
        for coord in coords:
            for element in kernel_element:
                if img[coord[0] + element[0]][coord[1] + element[1]] != 1:
                    dilated_img[coord[0] + element[0]][coord[1] + element[1]] = 1    
    except:
        pass

    return dilated_img

def erode_img(coords, img):
    eroded_img = img.copy()
    try:
        for coord in coords:
            for element in kernel_element:
                if img[coord[0] + element[0]][coord[1] + element[1]] != 1:
                    eroded_img[coord[0]][coord[1]] = 0    
                    break
    except:
        pass

    return eroded_img

def openning_img(coords, img):
    eroded = erode_img(coords, img)
    coords = map_coordinates(eroded)

    opened = dilate_img(coords, eroded)
    return opened

def closing_img(coords, img):
    dilated = dilate_img(coords, img)
    coords = map_coordinates(dilated)
    closed = erode_img(coords, dilated)

    return closed

def border_extraction(coords, img):
    eroded = erode_img(coords, img)
    border = img - eroded
    return border


def get_kernel(width, heigth):
    kernel = []
    for h in range(heigth):
        for w in range(width):
            kernel.append((h,w))
            
    return kernel

def get_number_of_iterations(coords, img):
    n = 0
    skel = np.zeros(img.shape, np.uint8)
    results = []
    # while sum(sum(img)) != 0:
    while n != 25:
        opened = openning_img(coords, img)
        result = img - opened
        eroded = erode_img(coords, img)
        
        skel = cv2.bitwise_or(skel,result)
        results.append(skel)
        img = eroded.copy()
        coords = map_coordinates(img)
        n += 1

    return results

def get_skeleton(coords, img):
    opened = openning_img(coords, img)
    result1 = img - opened
    
    coords = map_coordinates(opened)
    eroded = erode_img(coords, opened)
    return result1
    
kernel_element = get_kernel(8,8)
coords = map_coordinates(img)

n = get_number_of_iterations(coords, img)

dilated_img = dilate_img(coords, img)
eroded_img = erode_img(coords, img)

open_img = openning_img(coords, img)
close_img = closing_img(coords, img)

border = border_extraction(coords, img)

skeleton = get_skeleton(coords, img) 


plt.imshow(dilated_img, cmap='gray')
plt.imshow(eroded_img, cmap='gray')
plt.imshow(open_img, cmap='gray')
plt.imshow(close_img, cmap='gray')
plt.imshow(border, cmap='gray')
plt.imshow(skeleton, cmap='gray')

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.imshow(erosion, cmap='gray')
plt.imshow(dilation, cmap='gray')
plt.imshow(opening, cmap='gray')
plt.imshow(closing, cmap='gray')


coin_img = cv2.imread('/home/alvaro/Documentos/mestrado/PDI/coins.png', cv2.IMREAD_GRAYSCALE)
