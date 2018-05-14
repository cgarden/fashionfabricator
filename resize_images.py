import Image
from resizeimage import resizeimage
import glob

addrs = glob.glob('../img/*/*.jpg')

for addr in addrs:
	fd_img = open(addr,'r')
	# img = Image.open(fd_img)
	fd_img = resizeimage.resize_contain(fd_img, [64, 64])
	fd_img.convert('L')
	fd_img.save(addr, fd_img.format)
	fd_img.close()