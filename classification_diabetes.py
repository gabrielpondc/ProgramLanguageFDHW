import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
#print(x_data.shape, y_data.shape)
num_attributes = x_data.shape[1] # the number of samples
X = tf.placeholder(tf.float32, shape=[None, num_attributes])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([num_attributes, 1]), name='W')
b = tf.Variable(tf.random_normal([1]), name='b')

model = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-Y * tf.log(model) - (1 - Y) * tf.log(1 - model))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

predict = tf.cast(model > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

# Session
with tf.Session() as sess:
	# Training
	sess.run(tf.global_variables_initializer())
	for step in range(20001):
		cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
		if step % 2000 == 0:
			print(cost_val)
	# Testing
	m, p, a = sess.run([model, predict, accuracy], feed_dict={X: x_data, Y: y_data})
	print(m,p,a)
