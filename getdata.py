import h5py
filename = 'fashion.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
# print len(list(f.keys()))
a_group_key = list(f.keys())[0]

# Get the data
data = f[a_group_key]
print len(data[0]), data[0]
print len(data[0][0]), data[0][0]
print len(data[0][0][32]), data[0][0][32]
print data[0][0][32][32]