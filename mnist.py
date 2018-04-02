import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#grab the mnist data sets
mnist = input_data.read_data_sets("MNIST/DATA", one_hot=True)

# x is a placeholder that will receive the images we pass to the NN, this is a tensor
x = tf.placeholder(tf.float32, shape=[None, 784]) #convert image to 784 element vector
#10 element vector, contains the predicted probability of each digit (0-9) in data eg. [0.14,0,...,.7] in this case it would be a 9
yBar = tf.placeholder(tf.float32, [None,10])

#define the weight and bias
Weight = tf.Variable(tf.zeros([784,10]))
bias = tf.Variable(tf.zeros([10])) # 1 bias per digit, we initialize to 0.

result = tf.nn.softmax(tf.matmul(x,Weight)+bias)
#loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yBar, logits=result))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initliaze global variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, yBar:batch_ys})

correct_prediction = tf.equal(tf.argmax(result,1), tf.argmax(yBar,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, yBar:mnist.test.labels})
print("Test accuracy: {0}".format(test_accuracy * 100.00))
sess.close()