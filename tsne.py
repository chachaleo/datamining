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
target_names = []
target_names.append("apple")
target_names.append("flower")
target_names.append("microwave")

apples = QuickDrawDataGroup("apple")
donut = QuickDrawDataGroup("flower")
eye = QuickDrawDataGroup("microwave")

data_array_array = []
maxi=0
nombre = 100

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

    

print("3")
from sklearn.datasets import load_digits

digits = load_digits(n_class=6)

print(np.array(data_array_array))
print(target)
X, y = np.array(data_array_array), target
n_samples, n_features = X.shape
n_neighbors = 30

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))

for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((255, 255)), cmap=plt.cm.binary)
    ax.axis("off")
_ = fig.suptitle("A selection from the dataset", fontsize=16)
import numpy as np
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
color = 0

def plot_embedding(X, title, ax):
    X = MinMaxScaler().fit_transform(X)
    for digit in target_names:
        if(digit == target_names[0]): color = 0
        if(digit == target_names[1]): color = 1
        if(digit == target_names[2]): color = 2
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(color),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
    TSNE,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

embeddings = {
    "Random projection embedding": SparseRandomProjection(
        n_components=2, random_state=42
    ),
    "Truncated SVD embedding": TruncatedSVD(n_components=2),
    "Linear Discriminant Analysis embedding": LinearDiscriminantAnalysis(
        n_components=2
    ),
    "Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=2),
    "Standard LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="standard"
    ),
    "Modified LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="modified"
    ),
    "Hessian LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="hessian"
    ),
    "LTSA LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="ltsa"
    ),
    "MDS embedding": MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
    "Random Trees embedding": make_pipeline(
        RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
        TruncatedSVD(n_components=2),
    ),
    "Spectral embedding": SpectralEmbedding(
        n_components=2, random_state=0, eigen_solver="arpack"
    ),
    "t-SNE embeedding": TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
    ),
    "NCA embedding": NeighborhoodComponentsAnalysis(
        n_components=2, init="pca", random_state=0
    ),
}
from time import time

projections, timing = {}, {}
for name, transformer in embeddings.items():
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X

    print(f"Computing {name}...")
    start_time = time()

    projections[name] = transformer.fit_transform(data, y)
    timing[name] = time() - start_time

from itertools import zip_longest

fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(17, 24))

for name, ax in zip_longest(timing, axs.ravel()):
    if name is None:
        ax.axis("off")
        continue
    title = f"{name} (time {timing[name]:.3f}s)"
    print("\n\nDEBUG\n\n")
    print(name)
    print(title)
    print(projections[name])
    print(ax)
    plot_embedding(projections[name], title, ax)

plt.show()    