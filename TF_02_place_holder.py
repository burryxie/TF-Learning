import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

    
#function to add a neural network layer
def add_layer(input_data,in_size,out_size,activation=None):
    Weight = tf.Variable(tf.random_uniform([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    
    wx_plus_biases = tf.matmul(input_data,Weight)+ biases
    
    if activation == None:
        output_data = wx_plus_biases
    else:
        output_data = activation(wx_plus_biases)
    return(output_data)

x = np.linspace(-1,1,300)[:,np.newaxis]
y = np.square(x) + 0.8 + np.random.normal(size=x.shape)

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,20,activation = tf.nn.relu)
output = add_layer(l1,20,1,activation = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(output-ys),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


#plt.axis([0,10,0,6])
plt.ion()

losses =[]
for i in range(100):
    print(i)
    sess.run(train_step,feed_dict={xs:x,ys:y})
    #if i % 50 ==0:
     #   print(i,sess.run([i,loss],feed_dict={xs:x,ys:y}))
    current_loss =sess.run([train_step,loss],feed_dict={xs:x,ys:y})
    losses.append(current_loss)
    plt.plot(losses)
    plt.show()
    plt.pause(0.05)
w =sess.run([train_step,Weight],feed_dict={xs:x,ys:y})