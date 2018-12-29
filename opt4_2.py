# coding:utf-8
# 预测多少或预测少的影响一样
# 0：导入模块，生成数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23445
COST = 1
PROFIT = 9

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
[[-0.7850587]
 [ 1.5075649]] 

After 500 training steps, w1 is: 
[[1.0426469]
 [1.0201896]] 

After 1000 training steps, w1 is: 
[[1.0437704]
 [1.0173142]] 

After 1500 training steps, w1 is: 
[[1.044894 ]
 [1.0144387]] 

After 2000 training steps, w1 is: 
[[1.0491023]
 [1.019722 ]] 

After 2500 training steps, w1 is: 
[[1.0423932]
 [1.0162501]] 

After 3000 training steps, w1 is: 
[[1.0435168]
 [1.0133747]] 

After 3500 training steps, w1 is: 
[[1.0398924]
 [1.0180615]] 

After 4000 training steps, w1 is: 
[[1.041016 ]
 [1.0151861]] 

After 4500 training steps, w1 is: 
[[1.0452243]
 [1.0204693]] 

After 5000 training steps, w1 is: 
[[1.0463479]
 [1.0175939]] 

After 5500 training steps, w1 is: 
[[1.0427235]
 [1.0222807]] 

After 6000 training steps, w1 is: 
[[1.0438471]
 [1.0194052]] 

After 6500 training steps, w1 is: 
[[1.0449706]
 [1.0165298]] 

After 7000 training steps, w1 is: 
[[1.0460942]
 [1.0136544]] 

After 7500 training steps, w1 is: 
[[1.0424699]
 [1.0183412]] 

After 8000 training steps, w1 is: 
[[1.0435934]
 [1.0154657]] 

After 8500 training steps, w1 is: 
[[1.0478017]
 [1.020749 ]] 

After 9000 training steps, w1 is: 
[[1.0410926]
 [1.0172771]] 

After 9500 training steps, w1 is: 
[[1.0422162]
 [1.0144017]] 

After 10000 training steps, w1 is: 
[[1.0464245]
 [1.0196849]] 

After 10500 training steps, w1 is: 
[[1.0397154]
 [1.016213 ]] 

After 11000 training steps, w1 is: 
[[1.0439237]
 [1.0214963]] 

After 11500 training steps, w1 is: 
[[1.0450473]
 [1.0186208]] 

After 12000 training steps, w1 is: 
[[1.041423 ]
 [1.0233077]] 

After 12500 training steps, w1 is: 
[[1.0425465]
 [1.0204322]] 

After 13000 training steps, w1 is: 
[[1.04367  ]
 [1.0175568]] 

After 13500 training steps, w1 is: 
[[1.0447936]
 [1.0146813]] 

After 14000 training steps, w1 is: 
[[1.0490019]
 [1.0199646]] 

After 14500 training steps, w1 is: 
[[1.0422928]
 [1.0164927]] 

After 15000 training steps, w1 is: 
[[1.0434164]
 [1.0136173]] 

After 15500 training steps, w1 is: 
[[1.0397921]
 [1.0183041]] 

After 16000 training steps, w1 is: 
[[1.0409156]
 [1.0154287]] 

After 16500 training steps, w1 is: 
[[1.0451239]
 [1.0207119]] 

After 17000 training steps, w1 is: 
[[1.0462475]
 [1.0178365]] 

After 17500 training steps, w1 is: 
[[1.0426232]
 [1.0225233]] 

After 18000 training steps, w1 is: 
[[1.0437467]
 [1.0196478]] 

After 18500 training steps, w1 is: 
[[1.0448703]
 [1.0167724]] 

After 19000 training steps, w1 is: 
[[1.0459938]
 [1.013897 ]] 

After 19500 training steps, w1 is: 
[[1.0423695]
 [1.0185838]] 

Final w1 id: 
[[1.0423828]
 [1.015691 ]]
'''
