# 动量法
def sgd_momentum(params, vs, lr, mom, batch_size):
    for param, v in zip(params, vs):
        v[:] = mom * v + lr * param.grad / batch_size
        param[:] -= v


import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import numpy as np
import random
import sys
sys.path.append('..')
import utils

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3, 4]
true_b = 4.2
X = nd.random_normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(scale=1, shape=y.shape)

# 初始化模型参数


def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    vs = []
    for param in params:
        param.attach_grad()
        # 把速度项初始化为和参数形状相同的零张量
        vs.append(param.zeros_like())
    return params, vs


net = utils.linreg
squared_loss = utils.squared_loss


def optimize(batch_size, lr, mom, num_epochs, log_interval):
    [w, b], vs = init_params()
    y_vals = [squared_loss(net(X, w, b), y).mean().asnumpy()]
    print('batch_size', batch_size)
    for epoch in range(1, num_epochs + 1):
        # 学习率自我衰减
        if epoch > 2:
            lr *= .1
        for batch_i, (features, label) in enumerate(
                utils.data_iter(batch_size, num_examples, X, y)):
            with autograd.record():
                output = net(features, w, b)
                loss = squared_loss(output, label)
            loss.backward()
            sgd_momentum([w, b], vs, lr, mom, batch_size)
            if batch_i * batch_size % log_interval == 0:
                y_vals.append(squared_loss(net(X, w, b), y).mean().asnumpy())
        print('epoch %d, learning_rate %f, loss %.4e'
              % (epoch, lr, y_vals[-1]))
    # 为了便于打印，改变输出形状并转化成numpy数组
    print('w:', w.reshape((1, -1)).asnumpy(), 'b:', b.asscalar(), '\n')
    x_vals = np.linspace(0, num_epochs, len(y_vals), endpoint=True)
    utils.semilogy(x_vals, y_vals, 'epoch', 'loss')
