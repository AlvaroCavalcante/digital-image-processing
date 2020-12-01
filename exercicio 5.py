import cv2 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

img = np.array(Image.open('/home/alvaro/Downloads/original/horseshoe-6.gif'))

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

kernel_element = [(0,0), (0,1), (0,2)]

coords = map_coordinates(img)
dilated_img = dilate_img(coords, img)

plt.imshow(dilated_img, cmap='gray')
