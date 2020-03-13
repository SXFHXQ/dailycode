# Xavier均匀初始化
# fan_in输入节点数量，fan_out输出节点数量
# 均匀分布方差为(low+high)^2/12=2/(fan_in+fan_out)  所以区间为(low,high)
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))  # 由方差得到上下界
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# 构造降噪自编码器类
# 加性高斯噪声的自动编码器
class AdditiveGaussianNioseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        '''
        :param n_input: 输入节点数
        :param n_hidden: 隐层节点数
        :param transfer_function: 隐层激活函数
        :param optimizer: 优化器
        :param scale: 高斯噪声方差
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.training_scale = scale
        self.weights = dict()

        with tf.name_scope('RawInput'):
            self.x = tf.placeholder(tf.float32, [None, self.n_input])   #输入
        with tf.name_scope('NoiseAdder'):
            self.scale = tf.placeholder(tf.float32)
            self.noise = self.x + self.scale * tf.random_normal((n_input,)) 
        with tf.name_scope('Encoder'):    #编码
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weight1')
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='bias1')
            self.hidden = self.transfer(
                tf.add(tf.matmul(self.noise, self.weights['w1']), self.weights['b1']))
        with tf.name_scope('Reconstruction'):    #解码
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32), name='weight2')
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='bias2')
            self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),
                                         self.weights['b2'])
        # 损失
        # 重构信号和原始信号的误差平方和
        with tf.name_scope('Loss'):
            self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2))
        with tf.name_scope('Train'):
            self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('begin to run session...')

    # 在批次上训练模型
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x:X, self.scale:self.training_scale})
        return cost

    # 在给定的样本集合上计算损失（用于测试阶段）
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale:self.training_scale})

    # 返回自编码器隐层的输出结果，获得抽象后的高阶特征表示
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale:self.training_scale})

    # 将隐层的高阶特征作为输入，将其重构为原始输入数据
    def generate(self, hidden=None):
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

    # 整体运行一遍前向复原过程，包括提取高阶特征以及重构原始数据，输入原始数据，输出重构后的数据
    def reconstruction(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X, self.scale:self.training_scale})

    # 获取隐层的权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐层偏置
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

AGN_AE = AdditiveGaussianNioseAutoencoder(n_input=784, n_hidden=200,
                                          transfer_function=tf.nn.softplus,
                                          optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                          scale=0.01)
print('把计算图写事件文件，在TensorBoard里面查看')
writer = tf.summary.FileWriter(logdir='logs', graph=AGN_AE.sess.graph)
writer.close()

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

# 使用sklearn.preprocess的数据标准化操作（0均值标准差为1） 预处理数据
# 首先在训练集上估计均值与方差，然后将其作用到训练集和测试集
def standard_scale(X_train, X_test):
    preprocesser = prep.StandardScaler().fit(X_train) 
    X_train = preprocesser.transform(X_train)  
    X_test = preprocesser.transform(X_test)  
    return X_train, X_test

# 获取随机的block数据的函数：取一个从0到len(data)-batch_size的随机整数
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

# 使用标准化操作变换数据集
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 定义训练参数
n_samples = int(mnist.train.num_examples)  # 训练样本数
training_epochs = 20
batch_size = 128  #
display_step = 1  # 输出结果的间隔

# 训练过程，每一轮epoch训练开始时，将平均损失avg_cost设为0
# 计算总共需要的batch数量（也就是迭代次数），这里使用的是有放回抽样
# 所以不能保证每个样本被抽到并参与训练
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    # 每个批次
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)  # 获取batch
        cost = AGN_AE.partial_fit(batch_xs)  # 获取每个批次上的cost
        avg_cost += cost / batch_size  # cost累加
    avg_cost /=total_batch

    if epoch % display_step ==0:
        print('epoch: %04d, cost = %.9f' % (epoch+1, avg_cost))

# 计算测试集上的cost
print('Total cost:', str(AGN_AE.calc_total_cost(X_test)))