from PIL import Image
import csv
import numpy as np


class mdata:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels



def readData():

    #read image pixel vals in
    fname = "UPNG/mdb"
    mgrams = []

    for i in range(1,10):
        if i < 100:
            if i < 10:
                name = fname + "00" + str(i) + ".png"
            else:
                name = fname + "0" + str(i) + ".png"
        else:
            name = fname + str(i) + ".png"

        im = Image.open(name).load() #Can be many different formats.

        pixels = []
        for k in range(1024):
            for j in range(1024):
                pixels.append(im[k,j])

        mgrams.append(np.ndarray(shape=(1024,), buffer=np.array(pixels)))

    mgrams = np.ndarray(shape=(1,), buffer=np.array(mgrams))

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

    del labels[9:]

    return mdata(mgrams, np.ndarray(shape=(1,), buffer=np.array(labels)))




