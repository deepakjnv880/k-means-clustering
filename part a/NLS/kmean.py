import math
import random
import numpy as np
from matplotlib import pyplot as plt
K=3
allowed_error=0.0001

def read_dataset(filename):
    file_object=open(filename,'r')
    file_data=file_object.readlines()
    data_vectors=[]#list of list
    for line in file_data:
        data_vectors.append([float(x) for x in line.split()])
    return data_vectors

def myrandom(data_vector_set):
    # min_element_wise=list(map(min, zip(*data_vector_set)))
    # max_element_wise=list(map(max, zip(*data_vector_set)))
    k_random_element=[]
    for i in range(K):
        k_random_element.append(random.choice(data_vector_set))
    return k_random_element

def compute_distance(data_vector,mean_vector):
    # print(data_vector)
    # print('*************************')
    # print(mean_vector)
    return math.sqrt(pow((data_vector[0]-mean_vector[0]),2)+(pow((data_vector[1]-mean_vector[1]),2)))

def assign_cluster(data_vector,mean_vector_set):
    min_distance=1000
    min_index=-1
    for i in range(K):
        dist=compute_distance(data_vector,mean_vector_set[i])
        if min_distance>dist:
            min_distance=dist
            min_index=i
    return min_index,min_distance

def compute_mean_vector(kcluster):
    return [np.array(kcluster)[:,0].mean(),np.array(kcluster)[:,1].mean()]

def main():
     # mean_vector=[]
     data_vector_set=[]

     for i in range(1,4):
         filename='data/Class'+str(i)+'.txt'
         data_vector_set[i-1:i]=(read_dataset(filename))

     # print(data_vector_set)
     print("its size = ",len(data_vector_set))
     # first_time=True
     c=[[] for v in range(3)]
     kcluster=[[] for v in range(K)]
     mean_vector_set=myrandom(data_vector_set)
     new_distortion_value=100
     old_distortion_value=0
     while abs(new_distortion_value-old_distortion_value)>allowed_error:

         old_distortion_value=new_distortion_value
         new_distortion_value=0
         ########################EXPECTATION########################
         #####assigning the data_vector_set to K kcluster
         kcluster=[[] for v in range(K)]
         for data_vector in data_vector_set:
             min_index,min_distance=assign_cluster(data_vector,mean_vector_set)
             new_distortion_value+=min_distance
             kcluster[min_index].append(data_vector)
         ########################MAXIMIZATION########################
         for i in range(K):
             mean_vector_set[i]=compute_mean_vector(kcluster[i])

         print('old j = ',old_distortion_value,' and new j = ',new_distortion_value)
     print kcluster[0]
     print(len(kcluster[0]))
     #naming the x axis
     plt.xlabel('x - axis')
     # naming the y axis
     plt.ylabel('y - axis')
     for k in range(K):
         for data_vector in kcluster[k]:
             if k==0:
                 # print '=='
                 plt.plot(data_vector[0], data_vector[1],marker='o', markersize=1, color="red")
             elif k==1:
                 plt.plot(data_vector[0], data_vector[1],marker='o', markersize=1, color="green")
             elif k==2:
                 plt.plot(data_vector[0], data_vector[1],marker='o', markersize=1, color="blue")
     # plt.show()
     # plt.legend()
     plt.savefig('cluster.png')

if __name__ == '__main__':
    main()
    # print("frist")
