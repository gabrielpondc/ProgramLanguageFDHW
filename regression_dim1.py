import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]
W=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))

model = x_data*W+b;
cost = tf.reduce_mean(tf.square(model - y_data))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session();
sess.run(tf.global_variables_initializer())
for step in range(2001):
	sess.run(train)
	print( sess.run(cost), sess.run(W), sess.run(b))
