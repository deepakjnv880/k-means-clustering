import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
#manhattan distance is used as distance measure
error_allowed=60
K=7

def calculate_manhattan_distance(first_pixel,second_pixel):
    # print('####### ############ ',first_pixel,second_pixel[0])
    distance=0
    for i in range(3):
        distance=distance+abs(int(first_pixel[i])-int(second_pixel[i]))
    # print("hi am ended,distance",distance)
    return distance

def assign_cluster(image_pixel,k_mean):
    min_index=-1;min_distance=1000
    for i in range(K):
        distance=calculate_manhattan_distance(k_mean[i],image_pixel)
        if min_distance>distance:
            min_distance=distance
            min_index=i
    return min_distance,min_index



def assign_k_mean1(image):
    k_mean=[]
    for i in range(K):
        x=random.randrange(0, 300, 5)
        y=random.randrange(0, 300, 5)
        k_mean.append(image[x][y])
    return k_mean

def assign_k_mean2():
    k_mean=[]
    k_mean.append([255,255,51])
    k_mean.append([0,0,255])
    k_mean.append([0,255,0])
    k_mean.append([32,32,32])
    k_mean.append([153,153,0])
    k_mean.append([255,128,0])
    k_mean.append([255,255,170])
    return k_mean
    # k_mean.appned([0,0,255])

def main():
    fname = "input.jpg"
    image=cv.imread(fname)
    image = cv.resize(image, (300, 300))
    # print(((image)))
    # print(narray[0,].reshape(imgage.shape[0:2]))
    # print(np.shape(np.array(image)))

    k_mean=assign_k_mean1(image)
    # print(k_mean)
    new_distortion_value=0
    old_distortion_value=7000
    k_cluster=[[] for i in range(K)]
    while(error_allowed<abs(old_distortion_value-new_distortion_value)):
        old_distortion_value=new_distortion_value
        new_distortion_value=0
        ########## clustering ##############
        for i in range(300):
            for j in range(300):
                # print(image[i][j][0])
                min_distance,assigned_cluster=assign_cluster(image[i][j],k_mean)
                # print(assigned_cluster)
                k_cluster[assigned_cluster].append(image[i][j])
                new_distortion_value=new_distortion_value+min_distance

        ######### improving k means #########
        deepak=8
        for i in range(K):
            # print("this is my cluster: \n\n\n",k_cluster[i])
            k_mean[i]=np.mean(np.array(k_cluster[i]), axis=0)

        print('Error is ',abs(old_distortion_value-new_distortion_value))
    dim=300
    for x in range(K):
        # print('################ \n ',k_cluster[x],'\n#############')
        data = np.zeros((dim, dim, 3), dtype=np.uint8)
        for i in range(dim):
            for j in range(dim):
                data[i][j]=[255,255,255]
        count=0
        check=False
        for i in range(0,(dim),1):
            for j in range(0,(dim),1):
                # print(i,j)
                data[i][j]=list(np.array(k_cluster[x][count]))
                count=count+1
                if count==len(k_cluster[x]):
                    check=True
                    break
            if check:
                break
        img = Image.fromarray(data, 'RGB')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.save('k clustered pixel image/'+str(x+1)+'.jpg')
        # img.show()

if __name__ == '__main__':
    main()
