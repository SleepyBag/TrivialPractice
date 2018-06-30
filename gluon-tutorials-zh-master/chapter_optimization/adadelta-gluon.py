import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import sys
sys.path.append('..')
import utils

# 生成数据集。
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 线性回归模型。
net = nn.Sequential()
net.add(nn.Dense(1))

net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'adadelta', {'rho': 0.999})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=10, decay_epoch=None,
               log_interval=10, features=features, labels=labels, net=net)
