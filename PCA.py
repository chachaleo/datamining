from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure

import imutils
import cv2
import sklearn

from sklearn.model_selection import train_test_split

from quickdraw import QuickDrawDataGroup
from quickdraw import QuickDrawData
import matplotlib.pyplot as plt
import PIL 
import numpy as np
import csv

import imageio
from PIL import Image, ImageTk
from urllib import request
import PySimpleGUI as sg


img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
shouldbe = "apple"

dim = (255, 255)
  
# resize image
im = cv2.resize(img, dim)

im_array = np.zeros(255*255)

for i in range(255):
    for j in range(255):
        if(im[i][j][0]==0 and im[i][j][1]==0 and im[i][j][2]==0):
            #data_array.append([i,j])
            im_array[i*255+j] = 1
            #plt.plot(j,i,'ko')

#plt.show()





target = []
target_names = []
target_names.append("apple")
target_names.append("flower")
target_names.append("microwave")

apples = QuickDrawDataGroup("apple")
donut = QuickDrawDataGroup("flower")
eye = QuickDrawDataGroup("microwave")

data_array_array = []
maxi=0
nombre = 50

for apple in apples.drawings:
    target.append(1)
    maxi = maxi +1
    #data_array = []
    data_array = np.zeros((255*255))

    
    data = np.asarray(apple.image)
    for i in range(255):
        for j in range(255):
            if(data[i][j][0]==0 and data[i][j][1]==0 and data[i][j][2]==0):
                #data_array.append([i,j])
                data_array[i*255 + j] = 1

    data_array_array.append(data_array)
        
    if(maxi==nombre):
        break


data_array_array_donut = []
maxi_donut=0
print("1")

for donut in donut.drawings:
    target.append(2)
    maxi_donut = maxi_donut +1
    #data_array_donut = []
    data_array_donut = np.zeros((255*255))

    data_donut = np.asarray(donut.image)
    for i in range(255):
        for j in range(255):
            if(data_donut[i][j][0]==0 and data_donut[i][j][1]==0 and data_donut[i][j][2]==0):
                #data_array_donut.append([i,j])
                data_array_donut[i*255 + j] = 1

    data_array_array.append(data_array_donut)

    
    if(maxi_donut==nombre):
        break


data_array_array_eye = []
maxi_eye=0

print("2")
for eye in eye.drawings:
    target.append(3)
    maxi_eye = maxi_eye +1
    #data_array_eye = []
    data_array_eye = np.zeros((255*255))


    data_eye = np.asarray(eye.image)
    for i in range(255):
        for j in range(255):
            if(data_eye[i][j][0]==0 and data_eye[i][j][1]==0 and data_eye[i][j][2]==0):
                #data_array_eye.append([i,j])
                data_array_eye[i*255 + j] = 1

    data_array_array.append(data_array_eye)
    if(maxi_eye==nombre):
        break 

    

print("3")
 
data_array_array.append(im_array)
target.append(4)

#Import required modules
from sklearn.decomposition import PCA
 
pca = PCA(2) # we need 2 principal components.
converted_data = pca.fit_transform(data_array_array)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map , c = target)

plt.colorbar()

plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()


im_array = converted_data[-1,:]
converted_data = converted_data[:-1,:]
target = target[:-1]


(trainData, testData, trainLabels, testLabels) = train_test_split(converted_data,
	target, test_size=0.25, random_state=42)

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
	test_size=0.1, random_state=84)
print(trainData)
print(np.array(trainData))
print(trainLabels)
print("Test labels :")
print(testLabels)

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainData, trainLabels)

	# evaluate the model and update the accuracies list
	score = model.score(valData, valLabels)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
	accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

print(im_array.reshape(1, -1))

prediction = model.predict(im_array.reshape(1, -1))[0]

if(prediction==1) : word = "apple"
if(prediction==2) : word = "flower"
if(prediction==3) : word = "microwave"


	# show the prediction
print("Should be : {}".format(shouldbe))
print("Prediction: {}".format(word))

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))
