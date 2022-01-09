from quickdraw import QuickDrawDataGroup
from quickdraw import QuickDrawData
import matplotlib.pyplot as plt
import PIL 
import numpy as np
import csv




apples = QuickDrawDataGroup("apple")

data_array_array = []
maxi=0

for apple in apples.drawings:
    maxi = maxi +1
    data_array = np.zeros((255,255))

    plt.imshow(apple.image)
    data = np.asarray(apple.image)
    for i in range(255):
    	for j in range(255):
    		if(data[i][j][0]==0 and data[i][j][1]==data[i][j][0] and data[i][j][2]==data[i][j][0]):
    			data_array[i,j]=1

    
    print(maxi)
    data_array_array.append(data_array)
    if(maxi==100):
    	break

print(data_array_array)   	

'''with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["word", "drawing"])
    for i in range(len(data_array_array)):
    	writer.writerow(["apple", data_array_array[i]])
'''