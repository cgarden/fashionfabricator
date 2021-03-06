import random
import numpy
import theano
import fashion_model
import sys
import functools
import time
from scipy.ndimage import filters

def iterate_minibatches(dataset, batch_size=50):
    # print 'dataset before shuffle: ', dataset
    random.shuffle(dataset)
    for offset in xrange(0, len(dataset), batch_size):
        # print dataset
        s = min(batch_size, len(dataset) - offset)
        batch_fonts = numpy.zeros((s,), dtype=numpy.int32)
        batch_chars = numpy.zeros((s,), dtype=numpy.int32)
        # print 'batch_fonts: ',batch_fonts
        # print 'batch_chars: ', batch_chars
        batch_ds = numpy.zeros((s, wh), dtype=theano.config.floatX)
        # print batch_ds
        for z in xrange(s):
            # print 'dataset: ', dataset
            i, j = dataset[offset + z]
            batch_fonts[z] = i
            batch_chars[z] = j
            # print 'data: ', data
            # print 'data[0]: ', data[0]
            # print 'data[0][0]: ', data[0][0]
            m = filters.gaussian_filter(data[i][j], sigma=random.random()*1.0) # data augmentation
            batch_ds[z] = m.flatten() * 1. / 255

        yield s, 1.0 * offset / len(dataset), batch_fonts, batch_chars, batch_ds


        
def iterate_run(dataset, fn, tag):
    total_loss, total_reg, total_count = 0, 0, 0
    for s, progress, input_font, input_char, output in iterate_minibatches(dataset):
        t0 = time.time()
        loss, reg = fn(input_font, input_char, output)
        t = time.time() - t0
        total_loss += float(loss) * s
        total_reg += float(reg) * s
        total_count += s
        sys.stdout.write('%s: %6.2f%%, perf: %.6f + %.6f (last minibatch: %.6f + %.6f, %.3fs)\r' % (tag, 100.0 * progress, total_loss / total_count, total_reg / total_count, float(loss), float(reg), t))
        sys.stdout.flush()

    sys.stdout.write('\n')
    return total_loss / total_count


data = fashion_model.get_data()
print data
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]
fashion_model = fashion_model.Model(n, k, wh)
fashion_model.try_load()
train_fn_w_learning_rate = fashion_model.get_train_fn()
test_fn = fashion_model.get_test_fn()
run_fn = fashion_model.get_run_fn()
train_set, test_set = fashion_model.sets()

print 'training...'
# for learning_rate in [1.0, 0.3, 0.1, 0.03, 0.01]:
for learning_rate in [1.0]:
    epoch, last_loss = 0, float('inf')
    while True:
        print 'epoch', epoch, 'learning rate', learning_rate
        train_fn = functools.partial(train_fn_w_learning_rate, learning_rate)
        iterate_run(train_set, train_fn, 'train')
        loss = iterate_run(test_set, test_fn, 'test ')
        if loss > last_loss and epoch > 3:
            break # decrease learning rate
        last_loss = loss
        fashion_model.save()
        epoch += 1
