
# example of zoom image augmentation
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

img_file = '/Users/haophung/Google Drive (brianphungai@gmail.com)/car_dataset/Mitsubishi Xpander/4.xpander-slideshow-homepage2.jpg'

# load the image
img = load_img(img_file)
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
# datagen = ImageDataGenerator(zoom_range=0.2)
# datagen = ImageDataGenerator(rotation_range=40)
# datagen = ImageDataGenerator(shear_range=0.2)
# datagen = ImageDataGenerator(height_shift_range=0.2)
# datagen = ImageDataGenerator(width_shift_range=0.2)
datagen = ImageDataGenerator(brightness_range=[0.7, 1.1])



# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
