# coding:utf-8
# 设损失函数loss = (w + 1)^2，令w的初值是常数5.
# 反向传播函数就是求最优w，即求最小的loss对应的w值。
import tensorflow as tf

# 定义待优化参数w的初值赋值为5
w = tf.Variable(tf.constant(5, dtype = tf.float32))

# 定义损失函数loss
loss = tf.square(w + 1)

# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)

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
After 0 steps: w is -7.000000, loss is 36.000000.
After 1 steps: w is 5.000000, loss is 36.000000.
After 2 steps: w is -7.000000, loss is 36.000000.
After 3 steps: w is 5.000000, loss is 36.000000.
After 4 steps: w is -7.000000, loss is 36.000000.
After 5 steps: w is 5.000000, loss is 36.000000.
After 6 steps: w is -7.000000, loss is 36.000000.
After 7 steps: w is 5.000000, loss is 36.000000.
After 8 steps: w is -7.000000, loss is 36.000000.
After 9 steps: w is 5.000000, loss is 36.000000.
After 10 steps: w is -7.000000, loss is 36.000000.
After 11 steps: w is 5.000000, loss is 36.000000.
After 12 steps: w is -7.000000, loss is 36.000000.
After 13 steps: w is 5.000000, loss is 36.000000.
After 14 steps: w is -7.000000, loss is 36.000000.
After 15 steps: w is 5.000000, loss is 36.000000.
After 16 steps: w is -7.000000, loss is 36.000000.
After 17 steps: w is 5.000000, loss is 36.000000.
After 18 steps: w is -7.000000, loss is 36.000000.
After 19 steps: w is 5.000000, loss is 36.000000.
After 20 steps: w is -7.000000, loss is 36.000000.
After 21 steps: w is 5.000000, loss is 36.000000.
After 22 steps: w is -7.000000, loss is 36.000000.
After 23 steps: w is 5.000000, loss is 36.000000.
After 24 steps: w is -7.000000, loss is 36.000000.
After 25 steps: w is 5.000000, loss is 36.000000.
After 26 steps: w is -7.000000, loss is 36.000000.
After 27 steps: w is 5.000000, loss is 36.000000.
After 28 steps: w is -7.000000, loss is 36.000000.
After 29 steps: w is 5.000000, loss is 36.000000.
After 30 steps: w is -7.000000, loss is 36.000000.
After 31 steps: w is 5.000000, loss is 36.000000.
After 32 steps: w is -7.000000, loss is 36.000000.
After 33 steps: w is 5.000000, loss is 36.000000.
After 34 steps: w is -7.000000, loss is 36.000000.
After 35 steps: w is 5.000000, loss is 36.000000.
After 36 steps: w is -7.000000, loss is 36.000000.
After 37 steps: w is 5.000000, loss is 36.000000.
After 38 steps: w is -7.000000, loss is 36.000000.
After 39 steps: w is 5.000000, loss is 36.000000.
'''
