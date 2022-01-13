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

import tkinter
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
import time


from PIL import Image, ImageTk

from sklearn.decomposition import PCA

def ML(file_path):
    file_path
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    dim = (255, 255)
    im = cv2.resize(img, dim)
    im_array = np.zeros(255*255)

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
    
    print("--LOADING THE DATA--")
    
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
    
    
    print("--LOADING COMPLETED--")
    
    
    print("--PERFORMING PCA--")
    
    
     
    data_array_array.append(im_array)
    target.append(4)
    
    #Import required modules
    
     
    pca = PCA(2) # we need 2 principal components.
    converted_data = pca.fit_transform(data_array_array)
    
    print("--PCA COMPLETED--")
    
    
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize = (10,6))
    c_map = plt.cm.get_cmap('jet', 10)
    plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
                cmap = c_map , c = target)
    
    plt.colorbar()
    
    plt.xlabel('PC-1') , plt.ylabel('PC-2')
    #plt.show()
    print("--PCA COMPLETED--")
    
    
    im_array = converted_data[-1,:]
    converted_data = converted_data[:-1,:]
    target = target[:-1]
    print("--PERFORMING K-NEAREST NEIGHBOR--")
    
    (trainData, testData, trainLabels, testLabels) = train_test_split(converted_data,
        target, test_size=0.25, random_state=42)
    
    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
        test_size=0.1, random_state=84)
    
    
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
    
    print("--PERFORMING K-NEAREST NEIGHBOR--")
    
    prediction = model.predict(im_array.reshape(1, -1))[0]
    
    if(prediction==1) : word = "apple"
    if(prediction==2) : word = "flower"
    if(prediction==3) : word = "microwave"
    
    return word
    
    
ws = Tk()
ws.title('Pictionary')
ws.geometry('400x300') 
    


       
def open_file():
    file_path = askopenfile(mode='r', filetypes=[('Image Files', '*png')])
    if file_path is not None:
        pass  
    pb1 = Progressbar(
        ws, 
        orient=HORIZONTAL, 
        length=300, 
        mode='determinate'
        )
    pb1.grid(row=4, columnspan=3, pady=20)
    for i in range(5):
        ws.update_idletasks()
        pb1['value'] += 20
        time.sleep(1)
    pb1.destroy()
    image1 = Image.open(file_path.name)
    newsize = (150, 150)
    image1 = image1.resize(newsize)
    test = ImageTk.PhotoImage(image1)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.grid(row=4, columnspan=3, pady=10)
    word = ML(file_path.name)
    Label(ws, text=word, foreground='green').grid(row=5, columnspan=1, pady=10)


    
    
adhar = Label(
    ws, 
    text='Draw on of the categories : '
    )
adhar.grid(row=0, column=0, padx=10)

adharbtn = Label(
    ws, 
    text ='Apple, flower, microwave'
    ) 
adharbtn.grid(row=0, column=1)

dl = Label(
    ws, 
    text='Upload your picture '
    )
dl.grid(row=1, column=0, padx=10)

dlbtn = Button(
    ws, 
    text ='Choose File ', 
    command = lambda:open_file()
    ) 
dlbtn.grid(row=1, column=1)








ws.mainloop()