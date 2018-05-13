import h5py
from PIL import Image
import numpy as np
filename = 'fashion_grey.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
# print len(list(f.keys()))
a_group_key = list(f.keys())[0]

# Get the data
data = f[a_group_key]
print data
print data.shape
# print len(data[0]), data[0], type(data[0])
# print len(data[0][0]), data[0][0]
# print len(data[0][0][32]), data[0][0][32]
# print data[0][0][32][32]


# img = PIL.Image.open('../png_fashion_images')
# matrix = numpy.array(img.getdata())

img = Image.fromarray(data[50][0], 'L')
print data[50][0]
# img.save('my.png')
img.show()