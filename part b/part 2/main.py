import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import math
#manhattan distance+euclidean_distance is used as distance measure
error_allowed=1000
# image_dim=400
image_dim=300
K=10

def calculate_euclidean_distance(first_coordinate,second_coordinate):
    # print(data_vector)
    # print('####### ############ ',first_coordinate,second_coordinate)
    # print('*************************',math.sqrt(pow((first_coordinate[0]-second_coordinate[0]),2)+(pow((first_coordinate[1]-second_coordinate[1]),2))))
    # print(mean_vector)
    return math.sqrt(pow((first_coordinate[0]-second_coordinate[0]),2)+(pow((first_coordinate[1]-second_coordinate[1]),2)))
def calculate_manhattan_distance(first_pixel,second_pixel):
    # print('####### ############ ',first_pixel,second_pixel)
    distance=0
    for i in range(3):
        distance=distance+abs(int(first_pixel[i])-int(second_pixel[i]))
    # print("hi am ended,distance",distance)
    return distance

def assign_cluster(x,y,image_pixel,k_mean):
    min_index=-1;min_distance=100000
    for i in range(K):
        distance=calculate_manhattan_distance(k_mean[i][1],image_pixel)+calculate_euclidean_distance(k_mean[i][0],[x,y])
        if min_distance>distance:
            min_distance=distance
            min_index=i
    return min_distance,min_index



def assign_k_random_mean(image):
    k_mean=[]
    for i in range(K):
        x=random.randrange(0, image_dim, 5)
        y=random.randrange(0, image_dim, 5)
        k_mean.append([[x,y],(image[x][y])])
    return k_mean

def main():
    fname = "input.jpg"
    image=cv.imread(fname)
    # print(np.shape(np.array(image)))
    image = cv.resize(image, (image_dim, image_dim))
    # print(np.shape(np.array(image)))
    k_mean=assign_k_random_mean(image)
    # print("below is initial random k means : \n",k_mean,"\n################\n")
    new_distortion_value=0
    old_distortion_value=10000
    assigned_cluster = np.zeros((image_dim, image_dim), dtype=np.uint8)
    # print(assigned_cluster)
    while(error_allowed<abs(old_distortion_value-new_distortion_value)):
        old_distortion_value=new_distortion_value
        new_distortion_value=0
        ########## assgning cluster ##############
        for i in range(image_dim):
            for j in range(image_dim):
                min_distance,predicated_cluster=assign_cluster(i,j,image[i][j],k_mean)
                assigned_cluster[i][j]=predicated_cluster
                new_distortion_value=new_distortion_value+min_distance

        ######## improving k means #########
        number_of_element_in_cluster=[0 for i in range(K)]
        k_mean=[]
        for i in range(K):
            k_mean.append([[0,0],[0,0,0]])
        for i in range(image_dim):
            for j in range(image_dim):
                k_mean[assigned_cluster[i][j]][0][0]=k_mean[assigned_cluster[i][j]][0][0]+i
                k_mean[assigned_cluster[i][j]][0][1]=k_mean[assigned_cluster[i][j]][0][1]+j
                k_mean[assigned_cluster[i][j]][1][0]=k_mean[assigned_cluster[i][j]][1][0]+int(image[i][j][0])
                k_mean[assigned_cluster[i][j]][1][1]=k_mean[assigned_cluster[i][j]][1][1]+int(image[i][j][1])
                k_mean[assigned_cluster[i][j]][1][2]=k_mean[assigned_cluster[i][j]][1][2]+int(image[i][j][2])
                number_of_element_in_cluster[assigned_cluster[i][j]]+=1


        for i in range(K):
            # print("this is my cluster: \n\n\n",k_cluster[i])
            k_mean[i][0][0]/=number_of_element_in_cluster[i]
            k_mean[i][0][1]/=number_of_element_in_cluster[i]
            k_mean[i][1][0]/=number_of_element_in_cluster[i]
            k_mean[i][1][1]/=number_of_element_in_cluster[i]
            k_mean[i][1][2]/=number_of_element_in_cluster[i]

        print('Error is ',abs(old_distortion_value-new_distortion_value))



    # img.show()
    # print("======================= NOW ====================")
    # print(k_mean)
    data = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)
    for i in range(image_dim):
        for j in range(image_dim):
            data[i][j]=k_mean[assigned_cluster[i][j]][1]

    img = Image.fromarray(data, 'RGB')
    img.save('output.jpg')
    img = cv.imread('output.jpg')
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite('output.jpg', img)


if __name__ == '__main__':
    main()
