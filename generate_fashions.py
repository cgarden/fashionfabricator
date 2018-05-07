import random
import numpy
import theano
import fashion_model

m = fashion_model.Model(artificial_font=True)
m.try_load()
run_fn = m.get_run_fn()
W = m.get_font_embeddings()
cov = numpy.cov(W.T)

def generate_fashion():
    return numpy.random.multivariate_normal(mean=numpy.zeros(m.d), cov=cov)

def generate_input(n_fashions=10):
    fashions = [generate_fashion() for f in xrange(n_fashions)]
    for f in xrange(n_fashions):
        a, b = fashions[f], fashions[(f+1)%n_fashions]
        for p in numpy.linspace(0, 1, 10):
            print f, p
            batch_is = numpy.zeros((m.k, m.d), dtype=theano.config.floatX)
            batch_js = numpy.zeros((m.k,), dtype=numpy.int32)
            for z in xrange(m.k):
                batch_is[z] = a * (1-p) + b * p
                batch_js[z] = z

            yield batch_is, batch_js

print 'generating...'
frame = 0
for input_i, input_j in generate_input():
    img = fashion_model.draw_grid(run_fn(input_i, input_j))
    img.save('dress_%06d.png' % frame)
    frame += 1
