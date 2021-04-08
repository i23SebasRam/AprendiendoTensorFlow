import tensorflow as tf
sess = tf.Session()
a = tf.ones((2,2))
b = tf.matmul(a,a)
b.eval(Session=sess)