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

fashion_train_path = '../fashion_images_resized/*.jpg'
addrs = glob.glob(fashion_train_path)
# print addrs

blank = PIL.Image.new('RGB', (w0*4, h0*4), 255)

def read_clothing(addr):
    # dress = PIL.ImageFont.truetype(fn, min(w0, h0))

    # # We need to make sure we scale down the fonts but preserve the vertical alignment
    # min_ly = float('inf')
    # max_hy = float('-inf')
    # max_width = 0
    # imgs = []

    # for addr in addrs:
    #     print '...', char
    #     # Draw character
    #     img = PIL.Image.new(addr)
    #     draw = PIL.ImageDraw.Draw(img)
    #     draw.text((w0, h0), char, font=dress)

    #     # Get bounding box
    #     diff = PIL.ImageChops.difference(img, blank)
    #     lx, ly, hx, hy = diff.getbbox()
    #     min_ly = min(min_ly, ly)
    #     max_hy = max(max_hy, hy)
    #     max_width = max(max_width, hx - lx)
    #     imgs.append((lx, hx, img))

    # print 'crop dims:', max_hy - min_ly, max_width
    # scale_factor = min(1.0 * h / (max_hy - min_ly), 1.0 * w / max_width)
    data = []

    # for addr in addrs:
        # img = img.crop((lx, min_ly, hx, max_hy))

        # # Resize to smaller
        # new_width = (hx-lx) * scale_factor
        # new_height = (max_hy - min_ly) * scale_factor
        # img = img.resize((int(new_width), int(new_height)), PIL.Image.ANTIALIAS)

        # # Expand to square
        # img_sq = PIL.Image.new('L', (w, h), 255)
        # offset_x = (w - new_width)/2
        # offset_y = (h - new_height)/2
        # print offset_x, offset_y
        # img_sq.paste(img, (int(offset_x), int(offset_y)))

    img = PIL.Image.open(addr)

    # Convert to numpy array
    # print img.getdata()
    matrix = numpy.array(img.getdata())
    # print matrix
    matrix = 255 - matrix
    resized = numpy.resize(matrix,(64,64))
    # print resized
    data.append(resized)

    return numpy.array(data)


# def get_ttfs(d='scraper/fonts'):
#     for dirpath, dirname, filenames in os.walk(d):
#         for filename in filenames:
#             if filename.endswith('.ttf') or filename.endswith('.otf'):
#                 yield os.path.join(dirpath, filename)

                
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
