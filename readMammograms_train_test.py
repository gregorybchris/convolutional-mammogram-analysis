from PIL import Image
import csv
import numpy as np
from numpy import float32
from numpy import int64
import random

class mdata:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data   = test_data
        self.test_labels = test_labels


def readData():
    num_images = 151
    len = 50
    #read image pixel vals in
    fname = "UPNG/mdb"
    mgrams = []
    test_mgrams = []

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

        center = random.randrange(482,542)

        pixels = [im[k,j] for k in range(center-len, center + len) for j in range(center - len, center +len)]
        # mgrams.append(np.ndarray(shape=(1024*1024,), buffer=np.array(pixels), dtype=int))
        if i < num_images - 50:
            mgrams.append(pixels)
        else:
            test_mgrams.append(pixels)
    # print(len(mgrams))
    # print(len(mgrams[0]))

    #print(len(mgrams[0]))
    #print(len(test_mgrams))
    mgrams = np.ndarray(shape=(num_images - 1 - 50,100*100), buffer=np.array(mgrams), dtype=float32)
    test_mgrams = mgrams = np.ndarray(shape=(50,100*100), buffer=np.array(test_mgrams), dtype=float32)
        # print("read image %d" % i)

    #read label data
    #https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners
    f=open("labels.csv")
    labels = []
    for row in csv.reader(f, delimiter=' '):
        #N, B, M
        if row[3] == "N":
            #labels.append([1,0,0])
            labels.append(0)
        elif row[3] == "B":
            #labels.append([0,1,0])
            labels.append(1)
        else:
            #labels.append([0,0,1])
            labels.append(2)

    del labels[num_images:]
    train_labels = np.ndarray(shape=(num_images - 1 - 50,), buffer=np.array(labels[:num_images - 50]), dtype=int64)
    test_labels = np.ndarray(shape=(50,), buffer=np.array(labels[num_images - 50:]), dtype=int64)

    #print (len(labels[:num_images - 50]))
    #print (len(labels[num_images - 50:]))

    return mdata(mgrams, test_mgrams, train_labels, test_labels)




