import pickle              # import module first
import gzip
import tsv

# Different pickle files based on training iteration

# Trial #1
# file_read = '../f_model.pickle.gz'
# file_write = 'embedding_reordered.tsv'

# Trial #2
file_read = '../f_model2.pickle.gz'
# file_write = 'embedding_2.tsv'

file_write = 'embedding_grey_3.tsv'

f = gzip.open(file_read)   # 'r' for reading; can be omitted
mydict = pickle.load(f)         # load file content as mydict
f.close()                       

print mydict.keys()

print mydict['input_font_bottleneck.W']
print len(mydict['input_font_bottleneck.W'])
print mydict['input_font_bottleneck.W'][0]
print len(mydict['input_font_bottleneck.W'][0])


writer = tsv.TsvWriter(open(file_write, "w"))

for row in mydict['input_font_bottleneck.W']:
	writer.list_line(row)

writer.close()
