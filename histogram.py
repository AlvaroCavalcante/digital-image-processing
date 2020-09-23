from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('/home/alvaro/Documentos/mestrado/PDI/imagens/dog.jpeg').convert('LA')

image = np.asarray(img)
image = image[:,:,0]

hist_array= []
histogram = {}

for i in range(0,256):
    hist_array.append(str(i))
    hist_array.append(0)

def Convert(lst): 
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)} 
    return res_dct 

histogram = Convert(hist_array)

for row in image:
    
    for column in row:
        histogram[str(column)] = histogram[str(column)] + 1


n_pixels = image.shape[0] * image.shape[1]

plt.hist(histogram.values())
