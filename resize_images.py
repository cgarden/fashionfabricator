from PIL import Image
from resizeimage import resizeimage
import glob

addrs = glob.glob('../img/*/*.jpg')

for addr in addrs[:10]:
	fd_img = open(addr, 'r')
	fd_img.show()
	img = Image.open(fd_img)
	img = resizeimage.resize_contain(img, [64, 64])
	img.convert('L')
	img.save(addr, img.format)
	fd_img.close()