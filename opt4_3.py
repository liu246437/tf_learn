# coding:utf-8
# 预测多少或预测少的影响一样
# 0：导入模块，生成数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23445
COST = 9
PROFIT = 1

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 1:定义神经网络的输入、参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))
y = tf.matmul(x, w1)

# 2:定义损失函数及反向传播方法。
# 定义损失函数喂MSE，反向传播函数为梯度下降。
loss_mse = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 3:生成会话，训练STEPS轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 20000
	for i in range(STEPS):
		start = (i * BATCH_SIZE) % 32
		end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
		sess.run(train_step, feed_dict = {x: X[start : end], y_: Y_[start : end]})
		if i % 500 == 0:
			print 'After %d training steps, w1 is: ' % (i)
			print sess.run(w1), '\n'

	print 'Final w1 id: \n', sess.run(w1)

# 在本代码第2步中尝试其他反向传播方法，看对收敛速度的影响，把体会写到笔记中
'''
After 0 training steps, w1 is: 
[[-0.8096106]
 [ 1.478405 ]] 

After 500 training steps, w1 is: 
[[0.8381155]
 [1.0362918]] 

After 1000 training steps, w1 is: 
[[0.9340863]
 [0.9915031]] 

After 1500 training steps, w1 is: 
[[0.93175787]
 [0.98792267]] 

After 2000 training steps, w1 is: 
[[0.93456584]
 [0.98549265]] 

After 2500 training steps, w1 is: 
[[0.934081  ]
 [0.98962975]] 

After 3000 training steps, w1 is: 
[[0.93175256]
 [0.9860493 ]] 

After 3500 training steps, w1 is: 
[[0.93126774]
 [0.9901864 ]] 

After 4000 training steps, w1 is: 
[[0.9340757]
 [0.9877564]] 

After 4500 training steps, w1 is: 
[[0.9335909]
 [0.9918935]] 

After 5000 training steps, w1 is: 
[[0.93126243]
 [0.988313  ]] 

After 5500 training steps, w1 is: 
[[0.9340704]
 [0.985883 ]] 

After 6000 training steps, w1 is: 
[[0.9335856]
 [0.9900201]] 

After 6500 training steps, w1 is: 
[[0.93639356]
 [0.9875901 ]] 

After 7000 training steps, w1 is: 
[[0.9307723 ]
 [0.99057674]] 

After 7500 training steps, w1 is: 
[[0.9335803]
 [0.9881467]] 

After 8000 training steps, w1 is: 
[[0.93309546]
 [0.9922838 ]] 

After 8500 training steps, w1 is: 
[[0.93590343]
 [0.9898538 ]] 

After 9000 training steps, w1 is: 
[[0.933575  ]
 [0.98627335]] 

After 9500 training steps, w1 is: 
[[0.93309015]
 [0.99041045]] 

After 10000 training steps, w1 is: 
[[0.9358981]
 [0.9879804]] 

After 10500 training steps, w1 is: 
[[0.9354133]
 [0.9921175]] 

After 11000 training steps, w1 is: 
[[0.93308485]
 [0.9885371 ]] 

After 11500 training steps, w1 is: 
[[0.9358928 ]
 [0.98610705]] 

After 12000 training steps, w1 is: 
[[0.935408  ]
 [0.99024415]] 

After 12500 training steps, w1 is: 
[[0.93307954]
 [0.9866637 ]] 

After 13000 training steps, w1 is: 
[[0.9325947]
 [0.9908008]] 

After 13500 training steps, w1 is: 
[[0.9354027]
 [0.9883708]] 

After 14000 training steps, w1 is: 
[[0.93491787]
 [0.9925079 ]] 

After 14500 training steps, w1 is: 
[[0.9325894]
 [0.9889274]] 

After 15000 training steps, w1 is: 
[[0.9353974]
 [0.9864974]] 

After 15500 training steps, w1 is: 
[[0.93491256]
 [0.9906345 ]] 

After 16000 training steps, w1 is: 
[[0.9325841 ]
 [0.98705405]] 

After 16500 training steps, w1 is: 
[[0.9320993 ]
 [0.99119115]] 

After 17000 training steps, w1 is: 
[[0.93490726]
 [0.9887611 ]] 

After 17500 training steps, w1 is: 
[[0.93442243]
 [0.9928982 ]] 

After 18000 training steps, w1 is: 
[[0.932094 ]
 [0.9893178]] 

After 18500 training steps, w1 is: 
[[0.93490195]
 [0.98688775]] 

After 19000 training steps, w1 is: 
[[0.9344171 ]
 [0.99102485]] 

After 19500 training steps, w1 is: 
[[0.9320887]
 [0.9874444]] 

Final w1 id: 
[[0.929963 ]
 [0.9892421]]
'''
