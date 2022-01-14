# Imports

# Imports

import numpy as np
import pandas as pd
import imutils
import cv2
import sklearn
import seaborn as sns

%matplotlib inline
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import kneighbors_graph
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import tree

from skimage import exposure

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from quickdraw import QuickDrawDataGroup
from quickdraw import QuickDrawData

import PIL 
import csv
import imageio
from PIL import Image, ImageTk
from urllib import request
import PySimpleGUI as sg


# Loading Dataset

target = []
target_names = ["apple", "flower", "microwave"]

nombre = 50
data_array_array = []
    
print("--LOADING THE DATA--")

for i in target_names:
    dataset = QuickDrawDataGroup(i, max_drawings=nombre)
    name = target_names.index(i)+1
    maxi=0
    
    for d in dataset.drawings:
        target.append(name)
        maxi = maxi +1
        
        data_array = np.zeros((255*255))
        data = np.asarray(d.image)
        for i in range(255):
            for j in range(255):
                if(data[i][j][0]==0 and data[i][j][1]==0 and data[i][j][2]==0):
                    data_array[i*255 + j] = 1
    
        data_array_array.append(data_array)
            
        if(maxi==nombre):
            break

    print(name)
    
print("--LOADING COMPLETED--")
     
print("--PERFORMING PCA--")
  
pca = PCA(2) # we need 2 principal components.
converted_data = pca.fit_transform(data_array_array)
plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)

plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15, cmap = c_map , c = target)
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')

print("--PCA COMPLETED--")
    
im_array = converted_data[-1,:]
converted_data = converted_data[:-1,:]
target = target[:-1]
    
    
    
# training data 70%, test data 30%
(trainData, testData, trainLabels, testLabels) = train_test_split(converted_data, target, test_size=0.3, random_state=42)
    
# validation data: 10% of training data 
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=42)
    
# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels))) 

print("--------------------------------")
print("Method: k-nearest Neighbor")
print("--------------------------------")

# values of k for our k-Nearest Neighbor classifier 
kVals = range(1, 20, 1) 
accuracies = []

for k in kVals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# value of k with largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))


# --------------------------------
print('FIRST MODEL with holdout-method')

# re-train  classifier 
model = KNeighborsClassifier(n_neighbors=k)
model.fit(trainData, trainLabels)

predictions = model.predict(testData)

# classification report
print("EVALUATION")
print(classification_report(testLabels, predictions))



# --------------------------------
print('SECOND MODEL - with cross-validation')

# create and train model with cv of 5 
model = KNeighborsClassifier(n_neighbors=kVals[i])
cv_scores = cross_val_score(model, trainData, trainLabels, cv=5)

print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))


print("--------------------------------")
print("Method: Decision tree")
print("--------------------------------")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainData, trainLabels)

n_nodes = clf.tree_.node_count
print("Number of nodes :")
print(n_nodes)
print("Number of max depth :")
print(clf.tree_.max_depth)

pred = clf.predict(testData)

error_rate = 1 - accuracy_score(pred,testLabels)
print('Error rate',error_rate)
values = np.arange(1,15,1)

cv_scs = {}
for i in values:
    clf = tree.DecisionTreeClassifier(max_depth=i,random_state=0)
    cv_scs[i]= cross_val_score(clf, trainData, trainLabels, cv=5, n_jobs=-1)
    
error_score=[]

for i in cv_scs.values():
    error_score.append(1-i)
plt.plot(cv_scs.keys(),error_score)
plt.show()


print("--------------------------------")
print("Method: Random forest")
print("--------------------------------")

model = RandomForestClassifier(n_jobs=-1, random_state=0)
model.fit(trainData, trainLabels)
print(model)
y_pred_rf = model.predict(testData)
acc_rf = accuracy_score(testLabels, y_pred_rf)
print ('Random forest accuracy: ',acc_rf)

parameters = {'n_estimators': [10,20,40,60,80,100,120,140,160]}

clf_rf = RandomForestClassifier(n_jobs=-1, random_state=0)
rf = GridSearchCV(clf_rf, parameters, n_jobs=-1)
rf.fit(trainData, trainLabels)
y_pred_rf = rf.predict(testData)
acc_rf = accuracy_score(testLabels, y_pred_rf)
print ('Random forest accuracy: ',acc_rf)

results = pd.DataFrame(rf.cv_results_)
results.sort_values('mean_test_score', ascending = False)
print(results)

results.plot('param_n_estimators','mean_test_score')



print("--------------------------------")
print("Method: Support Vector Machine")
print("--------------------------------")


model = svm.SVC(kernel='linear')
model.fit(trainData,trainLabels)

y_pred=model.predict(testData)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(testLabels))
print(f"The model is {accuracy_score(y_pred,testLabels)*100}% accurate")

print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, y_pred))

y_true = np.array(testLabels)

print("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
cmdf = pd.DataFrame(cm, index=range(3), columns=range(3))

ax = sns.heatmap(cmdf, annot=True, cmap="Blues", fmt="g")
ax.set(ylabel="Actual", xlabel="Predicted")
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)

#--------------------
print("Cross Validation")
cv_scores = cross_val_score(model, trainData, trainLabels, cv=5)

print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))

print("--------------------------------")
print("Method: Multi layer perceptron")
print("--------------------------------")


clf_mlp = MLPClassifier(random_state=0)
clf_mlp.fit(trainData, trainLabels)
print(clf_mlp)
y_pred_mlp = clf_mlp.predict(testData)
acc_mlp = accuracy_score(testLabels, y_pred_mlp)
print ('mlp accuracy: ',acc_mlp)

parameters = {'hidden_layer_sizes' : [(50,), (100,), (784,), (50,50), (100,100), (784,784), (50,50,50), (100,100,100)] }

clf_mlp = MLPClassifier(random_state=0)
mlp = GridSearchCV(clf_mlp, parameters, n_jobs=-1)
mlp.fit(trainData, trainLabels)

results=pd.DataFrame(mlp.cv_results_)
results.sort_values('mean_test_score', ascending = False)



print("--------------------------------")
print("Method: Naive Bayes Classifier")
print("MISSING")
print("--------------------------------")


print("--------------------------------")
print("Method: Convolutional neuronal network")
print("NOT WORKING WITH PCA")
print("--------------------------------")

'''
# https://github.com/kradolfer/quickdraw-image-recognition/blob/master/quickdraw_image_recognition.ipynb

X_train = np.array(trainData) 
X_test = np.array(testData)
y_train = np.array(trainLabels)
y_test = np.array(testLabels)

X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 255, 255, 1).astype('float32')
X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 255, 255, 1).astype('float32')

num_pixels = 255*255
num_classes = 1

# GOOD GOOD
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(255, 255, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
'''


