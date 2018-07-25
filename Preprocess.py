import os
import numpy as np
from PIL import Image


path=os.getcwd()
listDir=os.listdir(path+"/Edit")
X=np.asarray([])
for i in listDir:
    link=path+"/Edit"+"/"+i
    img=np.asarray(Image.open(link).convert('LA'))
    print(img.shape)
    k=img.flatten()
    X=np.append(X,k)
    print(img.size)
X = np.reshape(X, (-1, 20000))
Y=np.zeros(495)
Y[0:45]=1
Y[45:90]=2
Y[90:135]=3
Y[135:180]=4
Y[180:225]=5
Y[225:270]=6
Y[270:315]=7
Y[315:360]=8
Y[360:405]=9
Y[405:450]=10
Y[450:495]=11
np.save('X',X)
np.save('Y',Y)

