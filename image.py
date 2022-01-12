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


img = cv2.imread('pomm.png', cv2.IMREAD_UNCHANGED)

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

apples = QuickDrawDataGroup("apple")

donut = QuickDrawDataGroup("flower")

eye = QuickDrawDataGroup("microwave")

data_array_array = []
maxi=0
nombre = 60

for apple in apples.drawings:
    target.append("apple")
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
    target.append("flower")
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
    target.append("microwave")
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

    

mnist = datasets.load_digits()
print("3")

#def shuffle(A):
#   for i in range(len(A)):
#       temp = A[i]
#       A[i] = A[(i+13*i)%len(A)]
#       A[(i+13*i)%len(A)] = temp
#   return A    
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

prediction = model.predict(im_array.reshape(1, -1))[0]

'''# Get one PNG file from website and save to file
url = (
    "https://raw.githubusercontent.com/PySimpleGUI/PySimpleGUI/master/"
    "images/for_readme/Logo%20with%20text%20for%20GitHub%20Top.png")
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36')}
req = request.Request(url, headers=headers)
response = request.urlopen(req)
if response.status != 200:
    print("Failed to load image from website !")
    exit()
data = response.read()'''

filename = "pomm.png"

im = Image.open(filename)

im = im.resize((150,150), resample=Image.BICUBIC)

sg.theme('DarkGreen3')

layout = [
    [sg.Text("Your beautifull draw congratulation it's so beautifull")],
    [sg.Image(size=(25, 25), key='-IMAGE-')],
    [sg.Text("Is it a beautifull : {}".format(prediction))],
]
window = sg.Window('Window Title', layout, margins=(20, 20), finalize=True)

# Convert im to ImageTk.PhotoImage after window finalized
image = ImageTk.PhotoImage(image=im)

# update image in sg.Image
window['-IMAGE-'].update(data=image)

while True:

    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

window.close()


'''
from PIL import Image, ImageTk
from urllib import request
import PySimpleGUI as sg

# Get one PNG file from website and save to file
url = (
    "https://raw.githubusercontent.com/PySimpleGUI/PySimpleGUI/master/"
    "images/for_readme/Logo%20with%20text%20for%20GitHub%20Top.png")
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36')}
req = request.Request(url, headers=headers)
response = request.urlopen(req)
if response.status != 200:
    print("Failed to load image from website !")
    exit()
data = response.read()

filename = "pomme.png"

im = Image.open(filename)

sg.theme('DarkGreen3')

layout = [
    [sg.Image(size=(255, 255), key='-IMAGE-')],
]
window = sg.Window('Window Title', layout, margins=(0, 0), finalize=True)

# Convert im to ImageTk.PhotoImage after window finalized
image = ImageTk.PhotoImage(image=im)

# update image in sg.Image
window['-IMAGE-'].update(data=image)

while True:

    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

window.close()'''