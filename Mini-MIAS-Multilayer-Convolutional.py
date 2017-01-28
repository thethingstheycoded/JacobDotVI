import re
import numpy
import os
import tensorflow as tf
from random import randint

def read_pgm(filename, byteorder='>'):
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
  images_tensor = numpy.zeros((num_images, 1024*1024))
  i = 0
  for dirName, subdirList, fileList in os.walk(image_dir):
    for fname in fileList:
        if fname.endswith(".pgm"):
          images_tensor[i] = read_pgm(image_dir+fname, byteorder='<')
          i += 1

  # Create a tensor for the labels
  labels_tensor = numpy.zeros((num_images, 7))
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


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Stride of 1 from MNIST
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# Pooling over 2x2 blocks, derived from MNIST
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


if __name__ == "__main__":

  # Import the data into numpy vectors
  images_train, labels_train = import_images("images/", 322)

  # Set up the model variables
  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, 1024*1024])
  y_ = tf.placeholder(tf.float32, shape=[None, 7])

  #First Convolutional Layer
  W_conv1 = weight_variable([5,5,1,32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1, 1024, 1024, 1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  #Second Convolutional Layer
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # Densely connected layer
  # By going through two 2x2 poolings, we have reduced the
  # the image from 1024x1024 to 512x512 and then to 256x256
  # http://stackoverflow.com/questions/36987641/how-image-is-reduced-to-7x7-by-tensorflow
  W_fc1 = weight_variable([256*256*64, 256])
  b_fc1 = bias_variable([256])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 256*256*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Readout Layer
  W_fc2 = weight_variable([256, 7])
  b_fc2 = bias_variable([7])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess.run(tf.initialize_all_variables())

  # Set up Eval batch
  print('Allocating eval batches ...')
  test_xs = numpy.zeros((31, 1024*1024))
  test_ys = numpy.zeros((31, 7))
  for i in range(31):
    test_xs[i] = images_train[i]
    test_ys[i] = labels_train[i]

  # Train
  batch_xs = numpy.zeros((10,1024*1024))
  batch_ys = numpy.zeros((10,7))

  print('Beginning Model Training ...')

  # 20 training steps, using 10 images each
  for i in range(30):
    #Create a batch of random images for training
    for j in range(10):
      k = randint(32,321)
      batch_xs[j] = images_train[k]
      batch_ys[j] = labels_train[k]

    # Run the eval batch through the model and see how we're doing
    if i%10 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: test_xs, y_: test_ys, keep_prob:1.0})
      print("step %d, accuracy: %g"%(i, train_accuracy))

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

  #Clear out batch sets to save memory
  batch_xs = batch_ys = []

  print('Running Model Evaluation ...')

  print(accuracy.eval(feed_dict={x: test_xs, y_: test_ys, keep_prob:1.0}))

