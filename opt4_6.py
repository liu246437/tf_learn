# coding:utf-8
# 设损失函数loss = (w + 1)^2，令w的初值是常数5.
# 反向传播函数就是求最优w，即求最小的loss对应的w值。
import tensorflow as tf

# 定义待优化参数w的初值赋值为5
w = tf.Variable(tf.constant(5, dtype = tf.float32))

# 定义损失函数loss
loss = tf.square(w + 1)

# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

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
After 0 steps: w is 4.998800, loss is 35.985600.
After 1 steps: w is 4.997600, loss is 35.971207.
After 2 steps: w is 4.996400, loss is 35.956818.
After 3 steps: w is 4.995201, loss is 35.942436.
After 4 steps: w is 4.994002, loss is 35.928059.
After 5 steps: w is 4.992803, loss is 35.913689.
After 6 steps: w is 4.991604, loss is 35.899323.
After 7 steps: w is 4.990406, loss is 35.884964.
After 8 steps: w is 4.989208, loss is 35.870609.
After 9 steps: w is 4.988010, loss is 35.856262.
After 10 steps: w is 4.986812, loss is 35.841919.
After 11 steps: w is 4.985615, loss is 35.827583.
After 12 steps: w is 4.984417, loss is 35.813251.
After 13 steps: w is 4.983221, loss is 35.798927.
After 14 steps: w is 4.982024, loss is 35.784607.
After 15 steps: w is 4.980827, loss is 35.770294.
After 16 steps: w is 4.979631, loss is 35.755985.
After 17 steps: w is 4.978435, loss is 35.741684.
After 18 steps: w is 4.977239, loss is 35.727386.
After 19 steps: w is 4.976044, loss is 35.713097.
After 20 steps: w is 4.974848, loss is 35.698811.
After 21 steps: w is 4.973653, loss is 35.684532.
After 22 steps: w is 4.972458, loss is 35.670258.
After 23 steps: w is 4.971264, loss is 35.655991.
After 24 steps: w is 4.970069, loss is 35.641727.
After 25 steps: w is 4.968875, loss is 35.627472.
After 26 steps: w is 4.967681, loss is 35.613220.
After 27 steps: w is 4.966488, loss is 35.598976.
After 28 steps: w is 4.965294, loss is 35.584736.
After 29 steps: w is 4.964101, loss is 35.570503.
After 30 steps: w is 4.962908, loss is 35.556274.
After 31 steps: w is 4.961716, loss is 35.542053.
After 32 steps: w is 4.960523, loss is 35.527836.
After 33 steps: w is 4.959331, loss is 35.513626.
After 34 steps: w is 4.958139, loss is 35.499420.
After 35 steps: w is 4.956947, loss is 35.485222.
After 36 steps: w is 4.955756, loss is 35.471027.
After 37 steps: w is 4.954565, loss is 35.456841.
After 38 steps: w is 4.953373, loss is 35.442654.
After 39 steps: w is 4.952183, loss is 35.428478.
'''
