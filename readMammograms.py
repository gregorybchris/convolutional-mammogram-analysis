from PIL import Image
import csv
import numpy as np
from numpy import float32

class mdata:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels



def readData():
    num_images = 2
    #read image pixel vals in
    fname = "UPNG/mdb"
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

        pixels = [im[k,j] for k in range(1024) for j in range(1024)]
        # mgrams.append(np.ndarray(shape=(1024*1024,), buffer=np.array(pixels), dtype=int))
        mgrams.append(pixels)
    # print(len(mgrams))
    # print(len(mgrams[0]))
    mgrams = np.ndarray(shape=(num_images - 1,1024*1024), buffer=np.array(mgrams), dtype=float32)

        # print("read image %d" % i)

    #read label data
    #https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners
    f=open("labels.csv")
    labels = []
    for row in csv.reader(f, delimiter=' '):
        #N, B, M
        if row[3] == "N":
            labels.append([1,0,0])
        elif row[3] == "B":
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])

    del labels[2:]

    return mdata(mgrams, np.ndarray(shape=(num_images - 1,3), buffer=np.array(labels), dtype=float32))




