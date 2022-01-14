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
from tkinter import *
from tkinter import ttk, colorchooser
from PIL import Image
from PIL import ImageDraw

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

class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        self.image=Image.new("RGB",(255,255),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        self.pb1=Progressbar(self.controls,orient=HORIZONTAL,length=200,mode='determinate')

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
            self.draw.line(((self.old_x,self.old_y),(e.x,e.y)),(0,0,0),width=5)
        self.old_x = e.x
        self.old_y = e.y
        print(self.c)

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None      

    def changeW(self,e): #change Width of pen through slider
        self.penwidth = e
    def start(self):
        filename = "temp.jpg"
        self.image.save(filename)
        Label(self.controls, text='This is a : ',font=('arial 12')).grid(row=2,column=0)
        self.pb1.grid(row=3, column=0)
        self.pb1['value'] += 50
        self.pb1['value'] += 50
        word = ML(filename)
        
        self.pb1.destroy()
        
        Label(self.controls, text=word,font=('arial 12')).grid(row=3,column=0)     


    def clear(self):        
        self.c.delete(ALL)

    def change_fg(self):  #changing the pen color
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  #changing the background color canvas
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Draw : apple or flower or microwave',font=('arial 12')).grid(row=0,column=0)
        
        Button(self.controls, text ='Start',command = self.start).grid(row=1,column=0)
        #self.slider = ttk.Scale(self.controls,from_= 5, to = 5,command=self.changeW,orient=VERTICAL)
        #self.slider.set(self.penwidth)
        #self.slider.grid(row=0,column=1,ipadx=30)
        self.controls.pack(side=LEFT)
        
        self.c = Canvas(self.master,width=255,height=255,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors',menu=colormenu)
        colormenu.add_command(label='Brush Color',command=self.change_fg)
        colormenu.add_command(label='Background Color',command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        
        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()
