import h5py, random, numpy
import PIL, PIL.Image
from scipy.ndimage import filters
import random
import glob


# hdf5_path = '../fashion_grey.hdf5'
# fashion_train_path = '../fashion_images_resized/*.jpg'
# addrs = glob.glob(fashion_train_path)
# print data.shape

f = h5py.File('fashion_grey.hdf5', 'r')
data = f['fashion']
print data.shape

# data_3c = numpy.array([numpy.array(PIL.Image.open(addr)) for addr in addrs])
# print data_3c.shape
# # print data_3c
# data_1c = numpy.array([ numpy.array( [numpy.array([ numpy.average(data_3c[i][j][k]) for k in range(data_3c.shape[2]) ]) for j in range(data_3c.shape[1])] ) for i in range(data_3c.shape[0])])
# print data_1c.shape
# print data_1c
# # print addrs
# hdf5_file = h5py.File(hdf5_path, mode='w')
# hdf5_file.create_dataset("train_img", data=data_1c)
# data = hdf5_file['train_img']
# # print data
# # print data.shape

# # f = h5py.File('fonts.hdf5', 'r')
# # data = f['fonts']
# # print data.shape

# i = random.randint(0, data.shape[0]-1)
# for z in xrange(1):
#     j = random.randint(0, data.shape[1]-1)
#     m = data[i][j]
#     m = filters.gaussian_filter(m, sigma=random.random()*1.0)
#     img = PIL.Image.fromarray(numpy.uint8(255-m))
#     img.show()

i = random.randint(0, data.shape[0]-1)
print 'i= ', i
for z in xrange(10):
    j = random.randint(0, data.shape[1]-1)
    m = data[i][j]
    # m = filters.gaussian_filter(m, sigma=random.random()*1.0)
    img = PIL.Image.fromarray(numpy.uint8(255 - m))
    img.show()



