# coding: utf-8
import sys, os
import numpy as np

# 親ディレクトリのファイルをインポートするための設定
#sys.path.append(os.pardir) #old
from pathlib import Path # new (since python3.4)
cwd = Path(os.path.dirname(os.path.realpath(__file__)))
pwd = cwd.parent
sys.path.insert(1, str(pwd)) # put this path at the beginning is rule of thumb
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
