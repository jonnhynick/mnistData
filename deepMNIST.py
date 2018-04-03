import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#create input object that reads the data from the MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

sess = tf.InteractiveSession()  

#placeholders for MNIST input data
x = tf.placeholder(tf.float32,shape=[None,784])
yBar = tf.placeholder(tf.float32, [None,10])

#Change MNIST input data from a vector into a 28x28 pixel grayscale value cube
x_image = tf.reshape(x,[-1,28,28,1], name = "x_image")

# we will use RELU as our activation function, returns 0 if value is less than 0, and the value if such value is greater than 0
#   Need to initialize values to small positive number and with some noise so RELU doesnt automatically set them as 0.

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Convolution and pooling to control overfitting.
def conv2d(x,Weight):
    return tf.nn.conv2d(x, Weight, strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#define layers.
#layer 1
Weight_conv1 = weight_variable([5,5,1,32])  #define size of filter, 1 input channel only since they are grayscale images.
Bias_conv1 = bias_variable([32])
#pass to RELU activation function.
h_conv1 = tf.nn.relu(conv2d(x_image,Weight_conv1)+Bias_conv1)
h_pool1 = max_pool_2x2(h_conv1) #run result through max_pool

#layer 2
Weight_conv2 = weight_variable([5,5,32,64])
Bias_conv2 = bias_variable([64])
#convolution layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, Weight_conv2)+Bias_conv2)
h_pool2 = max_pool_2x2(h_conv2) #7x7 image

#fully connected layers.
Weight_fullyConnected = weight_variable([7*7*64,1024])  #1024 Neurons
Bias_fullyConnected = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fullyConnected = tf.nn.relu(tf.matmul(h_pool2_flat, Weight_fullyConnected)+Bias_fullyConnected)
#we need to avoid "over training" our machine so it doesnt perform bad on 'real world' data.
#dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fullyConnected_drop = tf.nn.dropout(h_fullyConnected, keep_prob)

#readout layer
Weight_fullyConnected_2 = weight_variable([1024,10])
Bias_fullyConnected_2 = bias_variable([10])

#define the model 
y_conv = tf.matmul(h_fullyConnected_drop, Weight_fullyConnected_2) + Bias_fullyConnected_2

#loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels = yBar))

#loss optimization
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

#correct
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.arg_max(yBar,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

#train the model
import time
num_steps = 3000
display_every = 100

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], yBar: batch[1], keep_prob:0.5})

    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], yBar:batch[1], keep_prob:1.0})
        end_time = time.time()
        print("Step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i,end_time-start_time,train_accuracy*100))

end_time = time.time()
print("Total training time for {0} batches: {1: .2f} seconds".format(i+1, end_time-start_time))

print("Test accuracy {0:.3f}".format(accuracy.eval(feed_dict={x:mnist.test.images, yBar: mnist.test.labels, keep_prob: 1.0})*100.0))

sess.close()