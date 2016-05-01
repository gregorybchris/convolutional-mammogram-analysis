from PIL import Image
import csv
import numpy as np
from numpy import float32
from numpy import uint8
import random
import scipy.misc

class mdata:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

def readData():
    num_images = 323
    #read image pixel vals in
    fname = "data/combined_set_52/mdb"
    mgrams = []

    for i in range(1,num_images):
        if i < 100:
            if i < 10:
                name = fname + "00" + str(i) + ".png"
            else:
                name = fname + "0" + str(i) + ".png"
        else:
            name = fname + str(i) + ".png"

        im = Image.open(name).load() #Can be many different formats.
        # pixels = []
        # for k in range(1024):
        #     for j in range(1024):
        #         pixels.append(im[k,j])

        # center = random.randrange(482,542)
        pixels = [im[k,j]/256.0 for k in range(0, 48) for j in range(0, 48)]
        mgrams.append(pixels)

    print(len(mgrams))
    print(len(mgrams[0]))


    # mgrams = np.ndarray(shape=(num_images - 1,48*48), buffer=np.array(mgrams), dtype=float32)

    #read label data
    #https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners
    f=open("data/labels.csv")
    labels = []
    n = 0
    b = 0
    m = 0
    for row in csv.reader(f, delimiter=' '):
        #N, B, M
        if row[3] == "N":
            #labels.append([1,0,0])
            labels.append(0)
            n += 1
        elif row[3] == "B":
            #labels.append([0,1,0])
            labels.append(1)
            b += 1
        else:
            #labels.append([0,0,1])
            labels.append(2)
            m += 1
    # del labels[num_images:]
    print(n,b,m)
    # return mdata(mgrams, np.ndarray(shape=(num_images - 1,), buffer=np.array(labels), dtype=uint8))
    return mdata(mgrams, labels)




