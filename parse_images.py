import tensorflow as tf

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [29, 39])
  return image_resized

filenames_list = []
for i in range(150):
	filenames_list.append("../../../fashion_images_resized/dress"+str(i+1)+".jpg")

# A vector of filenames.
filenames = tf.constant(filenames_list)

# `labels[i]` is the label for the image in `filenames[i].
# labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(_parse_function)