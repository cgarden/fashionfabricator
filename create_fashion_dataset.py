import h5py
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps
import os
import random
import string
import numpy
import sys
import glob

w, h = 64, 64
w0, h0 = 256, 256

# fashion_train_path = '../fashion_images_resized/*.jpg'
# fashion_train_path = '../png_fashion_images/*.png'
# addrs = glob.glob(fashion_train_path)

# 150 Color Images
# greyscale = False
# path_write = 'fashion.hdf5'

#150 Greyscale Images
# greyscale = True
# path_write = 'fashion_grey.hdf5'

# Deepfashion Dataset
fashion_train_path = '../png_fashion_images/*.png'
addrs = glob.glob(fashion_train_path)

# print addrs

def read_clothing(addr, greyscale=True):
    print addr

    data = []


    img = PIL.Image.open(addr)

    # Convert to numpy array
    matrix = numpy.array(img.getdata())
    print matrix
    print matrix.shape
    matrix = 255 - matrix
    if greyscale:
        new_matrix = numpy.mean(matrix, axis=1)
        print "new matrix", new_matrix
        print new_matrix.shape
        new_new_matrix = numpy.resize(new_matrix,(w*h))
    else:
        new_new_matrix = matrix
    resized = numpy.resize(new_new_matrix,(h,w))
    print resized
    print resized.shape
    print '########'
    data.append(resized)

    return numpy.array(data)

f = h5py.File(path_write, 'w')

dset = f.create_dataset('fashion', (1, 1, h, w), chunks=(1, 1, h, w), maxshape=(None, 1, h, w), dtype='u1')

i = 0
for addr in addrs:
    # print addr
    # try:
    #     data = read_clothing(addr)
    # except: # IOError:
    #     print 'was not able to read', addr
    #     continue
    data = read_clothing(addr,greyscale)

    # print data.shape
    dset.resize((i+1, 1, h, w))
    dset[i] = data
    i += 1
    f.flush()

f.close()
