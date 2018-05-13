import h5py
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps
import os
import random
import string
import numpy
import sys
import glob

w, h = 64, 64*3
w0, h0 = 256, 256

fashion_train_path = '../fashion_images_resized/*.jpg'
addrs = glob.glob(fashion_train_path)
# print addrs

def read_clothing(addr):
    print addr

    data = []


    img = PIL.Image.open(addr)

    # Convert to numpy array
    matrix = numpy.array(img.getdata())
    print matrix
    matrix = 255 - matrix
    resized = numpy.resize(matrix,(64*3,64))
    print resized
    data.append(resized)

    return numpy.array(data)

# Images in standard order              
f = h5py.File('fashion.hdf5', 'w')

dset = f.create_dataset('fashion', (1, 1, h, w), chunks=(1, 1, h, w), maxshape=(None, 1, h, w), dtype='u1')

i = 0
for addr in addrs:
    # print addr
    # try:
    #     data = read_clothing(addr)
    # except: # IOError:
    #     print 'was not able to read', addr
    #     continue
    data = read_clothing(addr)

    # print data.shape
    dset.resize((i+1, 1, h, w))
    dset[i] = data
    i += 1
    f.flush()

f.close()
