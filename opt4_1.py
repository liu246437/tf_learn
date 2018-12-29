# coding:utf-8
# 预测多少或预测少的影响一样
# 0：导入模块，生成数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23445

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
loss_mse = tf.reduce_mean(tf.square(y_ - y))
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
[[-0.8107192]
 [ 1.4848609]] 

After 500 training steps, w1 is: 
[[-0.41704282]
 [ 1.6703597 ]] 

After 1000 training steps, w1 is: 
[[-0.15661955]
 [ 1.7472507 ]] 

After 1500 training steps, w1 is: 
[[0.02318887]
 [1.7631347 ]] 

After 2000 training steps, w1 is: 
[[0.15345451]
 [1.7456988 ]] 

After 2500 training steps, w1 is: 
[[0.25258714]
 [1.7109689 ]] 

After 3000 training steps, w1 is: 
[[0.33155587]
 [1.6681559 ]] 

After 3500 training steps, w1 is: 
[[0.3969526]
 [1.6224874]] 

After 4000 training steps, w1 is: 
[[0.45278972]
 [1.5768703 ]] 

After 4500 training steps, w1 is: 
[[0.50155395]
 [1.5328648 ]] 

After 5000 training steps, w1 is: 
[[0.5448258]
 [1.4912494]] 

After 5500 training steps, w1 is: 
[[0.58364606]
 [1.4523654 ]] 

After 6000 training steps, w1 is: 
[[0.6187275]
 [1.4162971]] 

After 6500 training steps, w1 is: 
[[0.6505833]
 [1.3829931]] 

After 7000 training steps, w1 is: 
[[0.67960054]
 [1.3523296 ]] 

After 7500 training steps, w1 is: 
[[0.7060853]
 [1.3241495]] 

After 8000 training steps, w1 is: 
[[0.73029125]
 [1.2982811 ]] 

After 8500 training steps, w1 is: 
[[0.752433 ]
 [1.2745525]] 

After 9000 training steps, w1 is: 
[[0.7726969]
 [1.2527972]] 

After 9500 training steps, w1 is: 
[[0.79124933]
 [1.2328564 ]] 

After 10000 training steps, w1 is: 
[[0.8082381]
 [1.2145824]] 

After 10500 training steps, w1 is: 
[[0.8237976]
 [1.1978384]] 

After 11000 training steps, w1 is: 
[[0.8380487]
 [1.182497 ]] 

After 11500 training steps, w1 is: 
[[0.851103 ]
 [1.1684408]] 

After 12000 training steps, w1 is: 
[[0.86306125]
 [1.1555638 ]] 

After 12500 training steps, w1 is: 
[[0.87401557]
 [1.1437675 ]] 

After 13000 training steps, w1 is: 
[[0.88405037]
 [1.1329602 ]] 

After 13500 training steps, w1 is: 
[[0.893243 ]
 [1.1230596]] 

After 14000 training steps, w1 is: 
[[0.90166426]
 [1.1139901 ]] 

After 14500 training steps, w1 is: 
[[0.90937877]
 [1.1056811 ]] 

After 15000 training steps, w1 is: 
[[0.91644585]
 [1.0980694 ]] 

After 15500 training steps, w1 is: 
[[0.92292005]
 [1.0910971 ]] 

After 16000 training steps, w1 is: 
[[0.92885095]
 [1.0847092 ]] 

After 16500 training steps, w1 is: 
[[0.93428415]
 [1.078857  ]] 

After 17000 training steps, w1 is: 
[[0.93926173]
 [1.0734965 ]] 

After 17500 training steps, w1 is: 
[[0.94382113]
 [1.0685855 ]] 

After 18000 training steps, w1 is: 
[[0.947998]
 [1.064087]] 

After 18500 training steps, w1 is: 
[[0.9518244]
 [1.0599648]] 

After 19000 training steps, w1 is: 
[[0.95532995]
 [1.0561894 ]] 

After 19500 training steps, w1 is: 
[[0.9585414]
 [1.0527298]] 

Final w1 id: 
[[0.9614939]
 [1.0495782]]
'''
