import tensorflow as tf
import numpy as np

x = np.random.rand(100).astype(np.float32)
y = x * 0.6 + 3

#create tensorflow structure
weights = tf.Variable(np.random.uniform(size=[1],low=-1.0,high=1.0))
biases = tf.Variable(tf.zeros([1]))

y_pred = weights * x + biases


loss = tf.reduce_mean(tf.square(y-y_pred))
optimizer = tf.train.GradientDescentOptimizer(0.04)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
#create tensorflow stucture

sess = tf.Session()
sess.run(init)
#sess.run(init)  #run the nn

for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(weights),sess.run(biases))

