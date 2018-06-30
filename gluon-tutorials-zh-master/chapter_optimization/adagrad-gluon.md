# Adagrad——使用Gluon


在Gluon里，使用Adagrad很方便，我们无需重新实现该算法。

首先，导入本节中实验所需的包。

```{.python .input}
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import sys
sys.path.append('..')
import utils
```

下面生成实验数据集并定义线性回归模型。

```{.python .input  n=1}
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
```

我们可以在Trainer中定义优化算法名称`adagrad`。以下实验分别重现了[“Adagrad——从零开始”](adagrad-scratch.md)一节中实验结果。

```{.python .input  n=3}
net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'adagrad',
                        {'learning_rate': 0.9})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=None,
               log_interval=10, features=features, labels=labels, net=net)
```

## 小结

* 使用Gluon的`Trainer`可以方便地使用Adagrad。

## 练习

* 尝试使用其他的初始学习率，结果有什么变化？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2274)


![](../img/qr_adagrad-gluon.svg)
