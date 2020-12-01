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

def get_skeleton(coords, img):
    n = 0
    skel = np.zeros(img.shape, np.uint8)
    results = []

    while n != 350:
        opened = openning_img(coords, img)
        result = img - opened
        eroded = erode_img(coords, img)
        
        skel = cv2.bitwise_or(skel,result)
        # results.append(skel)
        img = eroded.copy()
        coords = map_coordinates(img)
        n += 1

    return skel

def get_cv_skeleton(coords, img):
    skel = np.zeros(img.shape, np.uint8)

    element = np.array([[0,1,0], [1,1,1], [0,1,0]]).astype('uint8')
    n = 0
    while True:
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        n+= 1
        print('iterations', n)
        if cv2.countNonZero(img)==0:
            break

    return skel

# kernel_element = get_kernel(8,8)
kernel_element  = np.array([[0,1,0], [1,1,1], [0,1,0]]).astype('uint8')

coords = map_coordinates(img)

skeleton = get_skeleton(coords, img)
 
dilated_img = dilate_img(coords, img)
eroded_img = erode_img(coords, img)

open_img = openning_img(coords, img)
close_img = closing_img(coords, img)

border = border_extraction(coords, img)


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
