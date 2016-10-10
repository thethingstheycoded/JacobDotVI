import re
import numpy
import os
import tensorflow as tf
from random import randint

def read_pgm(filename, byteorder='>'):
  #PGM code adapted from http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
  """Return image data from a raw PGM file as a numpy array.

  Format specification: http://netpbm.sourceforge.net/doc/pgm.html

  """

  with open(filename, 'rb') as f:
    buffer = f.read()
  try:
    header, width, height, maxval = re.search(
	b"(^P5\s(?:\s*#.*[\r\n])*"
	b"(\d+)\s(?:\s*#.*[\r\n])*"
	b"(\d+)\s(?:\s*#.*[\r\n])*"
	b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
  except AttributeError:
    raise ValueError("Not a raw PGM file: '%s'" % filename)
  return numpy.frombuffer(buffer,
			   dtype='u1' if int(maxval) < 256 else byteorder+'u2',
			   count=int(width)*int(height),
			   offset=len(header)
			  ).reshape((int(height)*int(width)))


def import_images(image_dir, num_images):
  # We'll create an array to hold all of the images, each of which is 1024x1024
  images_tensor = numpy.zeros( (num_images, 1024*1024) )
  i = 0
  for dirName, subdirList, fileList in os.walk(image_dir):
    for fname in fileList:
        if fname.endswith(".pgm"):
          images_tensor[i] = read_pgm(image_dir+fname, byteorder='<')
          i += 1

  # Create a tensor for the labels
  labels_tensor = numpy.zeros( (num_images, 7) )
  f = open("data.txt", 'r')
  for line in f:

    # The first value in the line is the database ID
    # some values are duplicated so we have to use this as the key
    image_num = int(line.split()[0].replace("mdb", ""))-1

    # The third value in the line is the class of abnormality present
    abnormality = line.split()[2]

    # we are going to built a one-hot vector for each label,
    # using the abnormality of the mammogram
    if abnormality == "CALC":
      labels_tensor[image_num] = numpy.array([(1,0,0,0,0,0,0)])
    elif abnormality == "CIRC":
      labels_tensor[image_num] = numpy.array([(0,1,0,0,0,0,0)])
    elif abnormality == "SPIC":
      labels_tensor[image_num] = numpy.array([(0,0,1,0,0,0,0)])
    elif abnormality == "MISC":
      labels_tensor[image_num] = numpy.array([(0,0,0,1,0,0,0)])
    elif abnormality == "ARCH":
      labels_tensor[image_num] = numpy.array([(0,0,0,0,1,0,0)])
    elif abnormality == "ASYM":
      labels_tensor[image_num] = numpy.array([(0,0,0,0,0,1,0)])
    elif abnormality == "NORM":
      labels_tensor[image_num] = numpy.array([(0,0,0,0,0,0,1)])

  return images_tensor, labels_tensor

if __name__ == "__main__":

  # Import the data into numpy vectors
  # Images from http://peipa.essex.ac.uk/info/mias.html
  images, labels = import_images("images/", 322)

  # Adapted from https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html
  x = tf.placeholder(tf.float32, [None, 1024*1024])
  W = tf.Variable(tf.zeros([1024*1024, 7]))
  b = tf.Variable(tf.zeros([7]))

  y = tf.nn.softmax(tf.matmul(x, W) + b)

  y_ = tf.placeholder(tf.float32, [None, 7])
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  # Train
  batch_xs = numpy.zeros((100,1024*1024))
  batch_ys = numpy.zeros((100,7))

  for i in range(1000):
    #Create a batch of random images for training
    for i in range(100):
      j = randint(0,271)
      batch_xs[i] = images[j]
      batch_ys[i] = labels[j]

    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

  # Eval
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  test_xs = numpy.zeros((50, 1024*1024))
  test_ys = numpy.zeros((50,7))
  for i in range(50):
    test_xs[i] = images_train[272+i]
    test_ys[i] = labels_train[272+i]

  print(sess.run(accuracy, feed_dict={x:test_xs, y_:test_ys}))
