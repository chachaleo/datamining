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


target = []

apples = QuickDrawDataGroup("microwave")

donut = QuickDrawDataGroup("flower")

eye = QuickDrawDataGroup("star")

data_array_array = []
maxi=0
nombre = 100

for apple in apples.drawings:
    target.append("apple")
    maxi = maxi +1
    #data_array = []
    data_array = np.zeros((255*255))

    plt.imshow(apple.image)
    data = np.asarray(apple.image)
    for i in range(255):
    	for j in range(255):
    		if(data[i][j][0]==0 and data[i][j][1]==data[i][j][0] and data[i][j][2]==data[i][j][0]):
    			#data_array.append([i,j])
    			data_array[i*255 + j] = 1

    data_array_array.append(data_array)
    if(maxi==nombre):
    	break


data_array_array_donut = []
maxi_donut=0
print("1")

for donut in donut.drawings:
    target.append("donut")
    maxi_donut = maxi_donut +1
    #data_array_donut = []
    data_array_donut = np.zeros((255*255))

    plt.imshow(donut.image)
    data_donut = np.asarray(donut.image)
    for i in range(255):
    	for j in range(255):
    		if(data_donut[i][j][0]==0 and data_donut[i][j][1]==data_donut[i][j][0] and data_donut[i][j][2]==data_donut[i][j][0]):
    			#data_array_donut.append([i,j])
    			data_array_donut[i*255 + j] = 1

    data_array_array.append(data_array_donut)
    if(maxi_donut==nombre):
    	break


data_array_array_eye = []
maxi_eye=0

print("2")
for eye in eye.drawings:
    target.append("eye")
    maxi_eye = maxi_eye +1
    #data_array_eye = []
    data_array_eye = np.zeros((255*255))


    plt.imshow(eye.image)
    data_eye = np.asarray(eye.image)
    for i in range(255):
    	for j in range(255):
    		if(data_eye[i][j][0]==0 and data_eye[i][j][1]==0 and data_eye[i][j][2]==0):
    			#data_array_eye.append([i,j])
    			data_array_eye[i*255 + j] = 1

    data_array_array.append(data_array_eye)
    if(maxi_eye==nombre):
    	break 

   	

mnist = datasets.load_digits()
print("3")

#def shuffle(A):
#	for i in range(len(A)):
#		temp = A[i]
#		A[i] = A[(i+13*i)%len(A)]
#		A[(i+13*i)%len(A)] = temp
#	return A	
#
#data_array_array = shuffle(data_array_array)
#target = shuffle(target)	
#print(target)
# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(data_array_array,
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
kVals = range(1, 60, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 60, 2):
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

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))
'''
# loop over a few random digits
for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
	# grab the image and classify it
	image = testData[i]
	prediction = model.predict(image.reshape(1, -1))[0]

	# convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
	# then resize it to 32 x 32 pixels so we can see it better
	image = image.reshape((8, 8)).astype("uint8")
	image = exposure.rescale_intensity(image, out_range=(0, 255))
	image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

	# show the prediction
	print("I think that digit is: {}".format(prediction))
	cv2.imshow("Image", image)
	cv2.waitKey(0)	'''