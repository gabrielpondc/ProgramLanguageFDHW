import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))

model = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-Y * tf.log(model) - (1 - Y) * tf.log(1 - model))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

predict = tf.cast(model > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float32))

with tf.Session() as sess:
	
	sess.run(tf.global_variables_initializer())
	for step in range(2001):
		c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
		
	m, p, a = sess.run([model, predict, accuracy], feed_dict={X: x_data, Y: y_data})

	print(m,p,a)
