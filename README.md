# k-means-clustering
Classification and Segmentation with clustering

# Details
There are two folder part A and part B. Each folder has data folder,python file and its cluster image.
</br>
a) For the dataset of Assignment 1, perform classification using k-means clustering for the non-linearly seperable cases 
</br>
b) Perform k-means clustering based segmentation of the given images :
</br>
i) When using only pixel colour values as features
</br>
II) When using both pixel colour and location values as features
</br>
(in both cases, display the segmentation output as a colour image, with different colours assigned pixels belonging to different clusters, and same colours assigned to pixels belonging to the same cluster)

# Assumption 
Euclidean distance is used as measure of distance in Distortion calculation.<br />
For initial expectation of k mean in EM step any k random point from given data set has been taken.

# Package required
Install matplotlib and tkinter using below command:  

	pip install matplotlib
	sudo apt-get install python3-tk

# How to run
Go to the folder and run below command:

	python kmean.py
