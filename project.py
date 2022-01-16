import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quickdraw import QuickDrawDataGroup
from tkinter import *
import seaborn as sns

import sklearn 
from sklearn.manifold import TSNE

import cv2
import PIL 



def ML(file_path, target_names):

    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    im = cv2.resize(img, (255, 255))

    im_array = np.zeros(255*255)
    for i in range(255):
        for j in range(255):
            if(im[i][j][0]==0 and im[i][j][1]==0 and im[i][j][2]==0):
                im_array[i*255+j] = 1

    apples, donut, eye, sheep, banana = QuickDrawDataGroup(target_names[0]), QuickDrawDataGroup(target_names[1]), QuickDrawDataGroup(target_names[2]), QuickDrawDataGroup(target_names[3]), QuickDrawDataGroup(target_names[4])   

    target = []
    data_array_array = []   
    data_size, index = 50, 0
    
    print("Processing data")
    
    for apple, donut, eye, sheep, banana in zip(apples.drawings, donut.drawings, eye.drawings, sheep.drawings, banana.drawings):
        target.append(1)
        target.append(2)
        target.append(3)
        target.append(4)
        target.append(5)
        index += 1
        data_array_apple, data_array_donut, data_array_eye = np.zeros((255*255)), np.zeros((255*255)), np.zeros((255*255))
        data_array_sheep, data_array_banana = np.zeros((255*255)), np.zeros((255*255))
        data_apple, data_donut, data_eye = np.asarray(apple.image), np.asarray(donut.image), np.asarray(eye.image)
        data_sheep, data_banana = np.asarray(sheep.image), np.asarray(banana.image)

        for i in range(255):
            for j in range(255):
                if(data_apple[i][j][0]==0): data_array_apple[i*255 + j] = 1
                if(data_donut[i][j][0]==0): data_array_donut[i*255 + j] = 1  
                if(data_eye[i][j][0]==0): data_array_eye[i*255 + j] = 1 
                if(data_sheep[i][j][0]==0): data_array_sheep[i*255 + j] = 1  
                if(data_banana[i][j][0]==0): data_array_banana[i*255 + j] = 1

        data_array_array.append(data_array_apple)
        data_array_array.append(data_array_donut)
        data_array_array.append(data_array_eye)
        data_array_array.append(data_array_sheep)
        data_array_array.append(data_array_banana)
            
        if(index==data_size):
            break  
  
    print("Processing completed")
    print("Performing TSNE and PCA")
    
    data_array_array.append(im_array)
    target.append(4)
    
    train = np.array(data_array_array)
    train = sklearn.preprocessing.StandardScaler().fit_transform(train)
    tsne = TSNE(n_components = 2, random_state=0)
     
    pca = sklearn.decomposition.PCA(n_components=10)
    pca_res_50 = pca.fit_transform(train)
    tsne_res = tsne.fit_transform(pca_res_50)

      
    im_array = tsne_res[-1,:]
    converted_data = tsne_res[:-1,:]
    target = target[:-1]

    print("Performing k-nearest neighbors")
    
    (trainData, testData, trainLabels, testLabels) = sklearn.model_selection.train_test_split(converted_data,
        target, test_size=0.25, random_state=42)
    
    (trainData, valData, trainLabels, valLabels) = sklearn.model_selection.train_test_split(trainData, trainLabels,
        test_size=0.1, random_state=84)
    
    
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))
    
    kVals = range(1, 30, 2)
    accuracies = []
    
    for k in range(1, 30, 2):
        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)
    
        score = model.score(valData, valLabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)
    
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
        accuracies[i] * 100))
    
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    
    print("Performing prediction")
    
    prediction = model.predict(im_array.reshape(1, -1))[0]
    
    if(prediction==1) : word = target_names[0]
    if(prediction==2) : word = target_names[1]
    if(prediction==3) : word = target_names[2]
    if(prediction==4) : word = target_names[3]
    if(prediction==5) : word = target_names[4]
    
    return word

class main:
    def __init__(self,master):
        self.target_names = ["apple","flower","microwave","sheep","banana"]
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 4
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        self.image=PIL.Image.new("RGB",(255,255),(255,255,255))
        self.draw=PIL.ImageDraw.Draw(self.image)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
            self.draw.line(((self.old_x,self.old_y),(e.x,e.y)),(0,0,0),width=5)
        self.old_x = e.x
        self.old_y = e.y
    

    def reset(self,e):
        self.old_x = None
        self.old_y = None      


    def start(self):
        filename = "temp.png"
        self.image.save(filename)
        Label(self.controls, text='This is a : ',font=('arial 12')).grid(row=2,column=0)
        word = ML(filename, self.target_names)
        Label(self.controls, text=word,foreground='green',font=('arial 12')).grid(row=3,column=0)     


    def clear(self):        
        self.c.delete(ALL)
        self.image=PIL.Image.new("RGB",(255,255),(255,255,255))
        self.draw=PIL.ImageDraw.Draw(self.image)        
        Label(self.controls, text='                  ',font=('arial 12')).grid(row=2,column=0)
        Label(self.controls, text='                  ',font=('arial 12')).grid(row=3,column=0) 



    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Draw '+self.target_names[0]+' or '+self.target_names[1]+' or '+self.target_names[2]+' or '+self.target_names[3]+' or '+self.target_names[4],font=('arial 12')).grid(row=0,column=0)       
        Button(self.controls, text ='Start',command = self.start).grid(row=1,column=0)
        self.controls.pack(side=LEFT)       
        self.c = Canvas(self.master,width=255,height=255,bg=self.color_bg,)
        self.c.pack(fill=NONE,expand=False)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        
        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()
