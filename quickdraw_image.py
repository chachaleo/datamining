from quickdraw import QuickDrawDataGroup
from quickdraw import QuickDrawData
import matplotlib.pyplot as plt
#qd = QuickDrawData()

apples = QuickDrawDataGroup("apple")

for apple in apples.drawings:
    plt.imshow(apple.image)
    #apple.image.show()
    print(apple.name)
    print(apple.strokes)
    break 
