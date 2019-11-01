import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import math
#manhattan distance+euclidean_distance is used as distance measure
error_allowed=10
# image_dim=400
image_dim=300
K=7

def calculate_euclidean_distance(first_coordinate,second_coordinate):
    return math.sqrt(pow((int(first_coordinate[0])-int(second_coordinate[0])),2)+(pow((int(first_coordinate[1])-int(second_coordinate[1])),2)))

def calculate_manhattan_distance(first_pixel,second_pixel):
    distance=0
    for i in range(3):
        distance=distance+abs(int(first_pixel[i])-int(second_pixel[i]))
    return distance

def assign_cluster(image_pixel,k_mean,flag):
    min_index=-1;min_distance=100000
    for i in range(K):
        if flag==0:
            distance=calculate_euclidean_distance(k_mean[i],image_pixel)
        elif flag==1:
            distance=calculate_manhattan_distance(k_mean[i],image_pixel)
        if min_distance>distance:
            min_distance=distance
            min_index=i
    return min_distance,min_index

def assign_k_random_mean(image):
    k_mean=[]
    for i in range(K):
        x=random.randrange(0, image_dim, 5)
        y=random.randrange(0, image_dim, 5)
        k_mean.append(image[x][y])
    return k_mean

def method1():
    print("\n\n##################### Method1 Start #######################\n\n")
    fname = "input.jpg"
    image=cv.imread(fname)
    image = cv.resize(image, (image_dim, image_dim))
    k_mean=assign_k_random_mean(image)
    # print("below is initial random k means : \n",k_mean,"\n################\n")
    new_distortion_value=0
    old_distortion_value=100000
    assigned_cluster = np.zeros((image_dim, image_dim), dtype=np.uint8)
    # print(assigned_cluster)
    while(error_allowed<abs(old_distortion_value-new_distortion_value)):
        old_distortion_value=new_distortion_value
        new_distortion_value=0
        ########## assgning cluster ##############
        for i in range(image_dim):
            for j in range(image_dim):
                min_distance,predicated_cluster=assign_cluster(image[i][j],k_mean,0)
                assigned_cluster[i][j]=predicated_cluster
                new_distortion_value=new_distortion_value+min_distance

        ######## improving k means #########
        number_of_element_in_cluster=[0 for i in range(K)]
        k_mean=[]
        for i in range(K):
            k_mean.append([0,0,0])
        for i in range(image_dim):
            for j in range(image_dim):
                # k_mean[assigned_cluster[i][j]][0][0]=k_mean[assigned_cluster[i][j]][0][0]+i
                # k_mean[assigned_cluster[i][j]][0][1]=k_mean[assigned_cluster[i][j]][0][1]+j
                k_mean[assigned_cluster[i][j]][0]=k_mean[assigned_cluster[i][j]][0]+int(image[i][j][0])
                k_mean[assigned_cluster[i][j]][1]=k_mean[assigned_cluster[i][j]][1]+int(image[i][j][1])
                k_mean[assigned_cluster[i][j]][2]=k_mean[assigned_cluster[i][j]][2]+int(image[i][j][2])
                number_of_element_in_cluster[assigned_cluster[i][j]]+=1


        for i in range(K):
            k_mean[i][0]/=number_of_element_in_cluster[i]
            k_mean[i][1]/=number_of_element_in_cluster[i]
            k_mean[i][2]/=number_of_element_in_cluster[i]

        print('Error is ',abs(old_distortion_value-new_distortion_value))

    # print("======================= NOW ====================")
    data = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)
    for i in range(image_dim):
        for j in range(image_dim):
            data[i][j]=k_mean[assigned_cluster[i][j]]

    img = Image.fromarray(data, 'RGB')
    img.save('euclidean_output.jpg')
    img = cv.imread('euclidean_output.jpg')
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite('euclidean_output.jpg', img)
    print("\n\n##################### Method1 End #######################\n\n")

def method2():
    print("##################### Method2 Start #######################\n\n")
    fname = "input.jpg"
    image=cv.imread(fname)
    image = cv.resize(image, (image_dim, image_dim))
    k_mean=assign_k_random_mean(image)
    # print("below is initial random k means : \n",k_mean,"\n################\n")
    new_distortion_value=0
    old_distortion_value=100000
    assigned_cluster = np.zeros((image_dim, image_dim), dtype=np.uint8)
    # print(assigned_cluster)
    while(error_allowed<abs(old_distortion_value-new_distortion_value)):
        old_distortion_value=new_distortion_value
        new_distortion_value=0
        ########## assgning cluster ##############
        for i in range(image_dim):
            for j in range(image_dim):
                min_distance,predicated_cluster=assign_cluster(image[i][j],k_mean,1)
                assigned_cluster[i][j]=predicated_cluster
                new_distortion_value=new_distortion_value+min_distance

        ######## improving k means #########
        number_of_element_in_cluster=[0 for i in range(K)]
        k_mean=[]
        for i in range(K):
            k_mean.append([0,0,0])
        for i in range(image_dim):
            for j in range(image_dim):
                # k_mean[assigned_cluster[i][j]][0][0]=k_mean[assigned_cluster[i][j]][0][0]+i
                # k_mean[assigned_cluster[i][j]][0][1]=k_mean[assigned_cluster[i][j]][0][1]+j
                k_mean[assigned_cluster[i][j]][0]=k_mean[assigned_cluster[i][j]][0]+int(image[i][j][0])
                k_mean[assigned_cluster[i][j]][1]=k_mean[assigned_cluster[i][j]][1]+int(image[i][j][1])
                k_mean[assigned_cluster[i][j]][2]=k_mean[assigned_cluster[i][j]][2]+int(image[i][j][2])
                number_of_element_in_cluster[assigned_cluster[i][j]]+=1


        for i in range(K):
            k_mean[i][0]/=number_of_element_in_cluster[i]
            k_mean[i][1]/=number_of_element_in_cluster[i]
            k_mean[i][2]/=number_of_element_in_cluster[i]

        print('Error is ',abs(old_distortion_value-new_distortion_value))

    # print("======================= NOW ====================")
    data = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)
    for i in range(image_dim):
        for j in range(image_dim):
            data[i][j]=k_mean[assigned_cluster[i][j]]

    img = Image.fromarray(data, 'RGB')
    img.save('manhattan_output.jpg')
    img = cv.imread('manhattan_output.jpg')
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite('manhattan_output.jpg', img)
    print("\n\n##################### Method2 End #######################\n\n")

def main():
    method1()
    method2()



if __name__ == '__main__':
    main()
