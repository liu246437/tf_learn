# coding:utf-8
# 设损失函数loss = (w + 1)^2，令w的初值是常数5.
# 反向传播函数就是求最优w，即求最小的loss对应的w值。
import tensorflow as tf

# 定义待优化参数w的初值赋值为5
w = tf.Variable(tf.constant(5, dtype = tf.float32))

# 定义损失函数loss
loss = tf.square(w + 1)

# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 生成会话，训练40轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print 'After %s steps: w is %f, loss is %f.' % (i, w_val, loss_val)
'''
After 0 steps: w is 2.600000, loss is 12.959999.
After 1 steps: w is 1.160000, loss is 4.665599.
After 2 steps: w is 0.296000, loss is 1.679616.
After 3 steps: w is -0.222400, loss is 0.604662.
After 4 steps: w is -0.533440, loss is 0.217678.
After 5 steps: w is -0.720064, loss is 0.078364.
After 6 steps: w is -0.832038, loss is 0.028211.
After 7 steps: w is -0.899223, loss is 0.010156.
After 8 steps: w is -0.939534, loss is 0.003656.
After 9 steps: w is -0.963720, loss is 0.001316.
After 10 steps: w is -0.978232, loss is 0.000474.
After 11 steps: w is -0.986939, loss is 0.000171.
After 12 steps: w is -0.992164, loss is 0.000061.
After 13 steps: w is -0.995298, loss is 0.000022.
After 14 steps: w is -0.997179, loss is 0.000008.
After 15 steps: w is -0.998307, loss is 0.000003.
After 16 steps: w is -0.998984, loss is 0.000001.
After 17 steps: w is -0.999391, loss is 0.000000.
After 18 steps: w is -0.999634, loss is 0.000000.
After 19 steps: w is -0.999781, loss is 0.000000.
After 20 steps: w is -0.999868, loss is 0.000000.
After 21 steps: w is -0.999921, loss is 0.000000.
After 22 steps: w is -0.999953, loss is 0.000000.
After 23 steps: w is -0.999972, loss is 0.000000.
After 24 steps: w is -0.999983, loss is 0.000000.
After 25 steps: w is -0.999990, loss is 0.000000.
After 26 steps: w is -0.999994, loss is 0.000000.
After 27 steps: w is -0.999996, loss is 0.000000.
After 28 steps: w is -0.999998, loss is 0.000000.
After 29 steps: w is -0.999999, loss is 0.000000.
After 30 steps: w is -0.999999, loss is 0.000000.
After 31 steps: w is -1.000000, loss is 0.000000.
After 32 steps: w is -1.000000, loss is 0.000000.
After 33 steps: w is -1.000000, loss is 0.000000.
After 34 steps: w is -1.000000, loss is 0.000000.
After 35 steps: w is -1.000000, loss is 0.000000.
After 36 steps: w is -1.000000, loss is 0.000000.
After 37 steps: w is -1.000000, loss is 0.000000.
After 38 steps: w is -1.000000, loss is 0.000000.
After 39 steps: w is -1.000000, loss is 0.000000.
'''
