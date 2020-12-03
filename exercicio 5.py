import cv2 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

img = np.array(Image.open('/home/alvaro/Downloads/original/butterfly-1.gif'))

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
    for coord in coords:
        for element in kernel_element:
            if coord[0] + element[0] < img.shape[0] and coord[1] + element[1] < img.shape[1]:
                if img[coord[0] + element[0]][coord[1] + element[1]] != 1:
                    dilated_img[coord[0] + element[0]][coord[1] + element[1]] = 1    

    return dilated_img

def erode_img(coords, img):
    eroded_img = img.copy()
    for coord in coords:
        for element in kernel_element:
            if coord[0] + element[0] < img.shape[0] and coord[1] + element[1] < img.shape[1]:
                if img[coord[0] + element[0]][coord[1] + element[1]] != 1:
                    eroded_img[coord[0]][coord[1]] = 0    
                    break
                
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
            
    return np.array(kernel)

def get_skeleton(coords, img):
    n = 0
    skel = np.zeros(img.shape, np.uint8)

    while n != 50:
        opened = openning_img(coords, img)
        result = img - opened
        eroded = erode_img(coords, img)
        
        skel = cv2.bitwise_or(skel,result)
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

kernel_element = get_kernel(11,11)
# kernel_element  = np.array([[0,1,0], [1,1,1], [0,1,0]]).astype('uint8')

coords = map_coordinates(img)
# skeleton = get_skeleton(coords, img)


"""
skeleton = get_cv_skeleton(coords, img)
 
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

""" 

coin_img = cv2.imread('/home/alvaro/Documentos/mestrado/PDI/coins 2.png', cv2.IMREAD_GRAYSCALE)

image_blur = cv2.blur(coin_img,(5,5))


laplacian = cv2.Laplacian(image_blur,cv2.CV_64F)
sobelx = cv2.Sobel(dst,cv2.CV_64F,1,0,ksize=3)  # x
sobely = cv2.Sobel(dst,cv2.CV_64F,0,1,ksize=3)  # y

plt.imshow(image_blur, cmap='gray')
plt.imshow(sobelx, cmap='gray')
plt.imshow(sobely, cmap='gray')

def binarize_img(img, threshold):
    img[img < threshold] = 1
    img[img >= threshold] = 0
    return img

kernel_element = get_kernel(5,5)

contours, hierarchy = cv2.findContours(bin_coin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = cv2.drawContours(image_blur, contours, -1, (0,255,0), 3)

bin_coin = 1 - binarize_img(sobely.copy(), 180) # 100
plt.imshow(bin_coin, cmap='gray')

dst = cv2.medianBlur(coin_img, 5)
plt.imshow(dst, cmap='gray')


coords = map_coordinates(bin_coin)

final_img = dilate_img(coords, bin_coin)
coords = map_coordinates(final_img)

plt.imshow(final_img, cmap='gray')

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(bin_coin,kernel,iterations = 2)
dilation = cv2.dilate(bin_coin,kernel,iterations = 1)
opening = cv2.morphologyEx(bin_coin, cv2.MORPH_OPEN, kernel,iterations = 1)
closing = cv2.morphologyEx(bin_coin, cv2.MORPH_CLOSE, kernel,iterations = 1)

plt.imshow(dilation, cmap='gray')
plt.imshow(erosion, cmap='gray')
plt.imshow(opening, cmap='gray')
plt.imshow(closing, cmap='gray')

final_img = erosion

for i in range(1):
    coords = map_coordinates(final_img)
    final_img = erode_img(coords, final_img)

plt.imshow(final_img, cmap='gray')


def is_border(img, l, c):
    border = [img[l + 1][c], img[l + -1][c],
              img[l][c + 1], img[l][c - 1]]
    
    if 0 not in border:
        return False
    return True

def pad_perimeter(img, first_line, first_col, last_line, last_col, figure_n):
    for l in range(last_line - first_line):
        for c in range(last_col - first_col):
            if img[first_line + l][first_col + c] == 1:
                img[first_line + l][first_col + c] = figure_n
    
def get_figure_perimeter(img, fig_area, figure_n):
    first_line = 9999
    last_line = 0
    first_col = 9999
    last_col = 0
    
    for i in range(len(fig_area)):
        first_line = fig_area[i][0] if fig_area[i][0] < first_line else first_line
        last_line = fig_area[i][0] if fig_area[i][0] > last_line else last_line
        first_col = fig_area[i][1] if fig_area[i][1] < first_col else first_col
        last_col = fig_area[i][1] if fig_area[i][1] > last_col else last_col
        
    return pad_perimeter(img, first_line, first_col, last_line, last_col, figure_n)

    
def get_figure_border(img, l, c, figure_n, fig_area):
    fig_area.append((l,c))
    if img[l][c+1] == 1 and is_border(img, l, c+1):
        img[l][c+1] = figure_n
        get_figure_border(img, l, c+1, figure_n, fig_area)
    elif img[l][c-1] == 1 and is_border(img, l, c-1):
        img[l][c-1] = figure_n
        get_figure_border(img, l, c-1, figure_n, fig_area)
    elif img[l+1][c] == 1 and is_border(img, l+1, c):
        img[l+1][c] = figure_n
        get_figure_border(img, l+1, c, figure_n, fig_area)
    elif img[l-1][c] == 1 and is_border(img, l-1, c):
        img[l-1][c] = figure_n
        get_figure_border(img, l-1, c, figure_n, fig_area)
    elif img[l+1][c-1] == 1 and is_border(img, l+1, c-1):
        img[l+1][c-1] = figure_n
        get_figure_border(img, l+1, c-1, figure_n, fig_area)
    elif img[l-1][c-1] == 1 and is_border(img, l-1, c-1):
        img[l-1][c-1] = figure_n
        get_figure_border(img, l-1, c-1, figure_n, fig_area)
    elif img[l-1][c+1] == 1 and is_border(img, l-1, c+1):
        img[l-1][c+1] = figure_n
        get_figure_border(img, l-1, c+1, figure_n, fig_area)
    elif img[l+1][c+1] == 1 and is_border(img, l+1, c+1):
        img[l+1][c+1] = figure_n
        get_figure_border(img, l+1, c+1, figure_n, fig_area)

    return get_figure_perimeter(img, fig_area, figure_n)

def get_borders():
    for l in range(final_img.shape[0]):
        for c in range(final_img.shape[1]):
            if final_img[l][c] == 1 and is_border(final_img, l, c):
                final_img[l][c] = 165

figures = 10
for l in range(final_img.shape[0]):
    for c in range(final_img.shape[1]):
        if final_img[l][c] == 1:
            figures+= 10
            final_img[l][c] = figures
            get_figure_border(final_img, l, c, figures, [])


unique_values = np.unique(final_img)
print(len(unique_values) - 1)
            
plt.imshow(final_img, cmap='gray')
