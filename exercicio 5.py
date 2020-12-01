import cv2 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

img = np.array(Image.open('/home/alvaro/Downloads/original/jar-16.gif'))

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
    opened = dilate_img(coords, img)
    coords = map_coordinates(opened)
    eroded = erode_img(coords, opened)

    return eroded

kernel_element = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
                  (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
                  (2,0), (2,1), (2,2), (2,3), (2,4), (2,5),
                  (3,0), (3,1), (3,2), (3,3), (3,4), (3,5),
                  (4,0), (4,1), (4,2), (4,3), (4,4), (4,5)]

coords = map_coordinates(img)
dilated_img = dilate_img(coords, img)
eroded_img = erode_img(coords, img)

open_img = openning_img(coords, img)
close_img = closing_img(coords, img)


plt.imshow(dilated_img, cmap='gray')
plt.imshow(eroded_img, cmap='gray')
plt.imshow(open_img, cmap='gray')
plt.imshow(close_img, cmap='gray')


kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.imshow(erosion, cmap='gray')
plt.imshow(dilation, cmap='gray')
plt.imshow(opening, cmap='gray')
plt.imshow(closing, cmap='gray')


