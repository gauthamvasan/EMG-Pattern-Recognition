import tensorflow as tf


x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(5*x*x - 3*x + 15, name='y')


model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))