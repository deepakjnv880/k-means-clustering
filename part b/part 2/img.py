import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import math

fname = "input.jpg"
image=cv.imread(fname)
# print(np.shape(np.array(image)))
image = cv.resize(image, (image_dim, image_dim))
data = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)
for i in range(image_dim):
    for j in range(image_dim):
        data[i][j][0]=int(image[i][j][0])
        data[i][j][1]=int(image[i][j][1])
        data[i][j][2]=int(image[i][j][2])



for x in range(K):
    for i in range(k_cluster_min_max[x][0][0],k_cluster_min_max[x][1][0],1):
        data[i][k_cluster_min_max[x][0][1]]=[0,0,0]
        data[i][k_cluster_min_max[x][1][1]]=[0,0,0]
    for i in range(k_cluster_min_max[x][0][1],k_cluster_min_max[x][1][1],1):
        data[k_cluster_min_max[x][0][0]][i]=[0,0,0]
        data[k_cluster_min_max[x][1][0]][i]=[0,0,0]


img = Image.fromarray(data, 'RGB')
img.save('output.jpg')
img.show()
