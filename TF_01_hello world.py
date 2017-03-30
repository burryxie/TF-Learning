import tensorflow as tf
import numpy as np

x = np.random.rand(1000).astype(np.float32)
y = 0.5 * (x^2) + x * 0.3 + 0.1

w1 = tf.Variable(tf.random_uniform([1], -1, 1))
w2 = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))

y_ = w1*x^2+w2*x+b
loss = tf.reduce_mean(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()


sess =tf.Session()
sess.run(init)

for i in range(1,20000):
    sess.run(train)
    if i % 50 == 0:
        print(i,sess.run(w1), sess.run(w2),sess.run(b))
        