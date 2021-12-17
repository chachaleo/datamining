#!pip install opencv-python
import cv2 
from quickdraw import QuickDrawDataGroup
from quickdraw import QuickDrawData
import matplotlib.pyplot as plt
import seaborn as sns
qd = QuickDrawData()

apples = QuickDrawDataGroup("apple")

for anvil in apples.drawings:
    plt.imshow(anvil.image)
    #anvil.image.show()
    print(anvil.name)
    print(anvil.strokes)
    break 