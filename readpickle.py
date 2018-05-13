import pickle              # import module first
import gzip
import tsv

f = gzip.open('../f_model.pickle.gz')   # 'r' for reading; can be omitted
mydict = pickle.load(f)         # load file content as mydict
# data = []
# for i in range(150):
# 	emb = []
# 	for j in range(40):
# 		emb.append(0.0)
# 	data.append(emb)
# mydict = {'input_fashion_bottleneck.W':data}
# pickle.dump(mydict, f, -1)
f.close()                       

print mydict.keys()

print mydict['input_font_bottleneck.W']
print len(mydict['input_font_bottleneck.W'])
print mydict['input_font_bottleneck.W'][0]
print len(mydict['input_font_bottleneck.W'][0])

# print mydict['output_sigmoid.b']
# print len(mydict['output_sigmoid.b'])

# print mydict['output_sigmoid.W']
# print len(mydict['output_sigmoid.W'])

# with open("embeddings.tsv", "w") as record_file:
# 	record_file.write(mydict['input_font_bottleneck.W'])

# ls = [['www.google.com','9','1'],['www.foo.com','177','43432'],['http://www.test.com','2132','4567'],['http://www.stackoverflow.com','8','9']]
# file = open('embeddings.tsv', 'w');
# writer = csv.writer(file)
# for row in mydict['input_font_bottleneck.W']:
# 	for val in row:
# 		writer.write
#     # writer.writerow(item)
# file.close()


# writer = tsv.TsvWriter(open("file.tsv", "w"))

# writer.comment("This is a comment")
# writer.line("Column 1", "Column 2", 12345)
# writer.list_line(["Column 1", "Column 2"] + list(range(10)))
# writer.close()

writer = tsv.TsvWriter(open("embedding_reordered.tsv", "w"))

for row in mydict['input_font_bottleneck.W']:
	writer.list_line(row)

writer.close()
