#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import numpy
import theano
import random
from theano import tensor as T

class LogisticLayer:
    def __init__(self, dim, learning_rate = 0.2):
        self.__dim = dim
        self.__learning_rate = learning_rate

        x = T.fvector('x')

        b = theano.shared(
            value=numpy.random.rand(1).astype(numpy.float32) - [numpy.float32(0.5)],
            name='b',
            borrow=True
        )


        w = theano.shared(
            value=numpy.random.rand(dim).astype(numpy.float32) - [numpy.float32(0.5)]*dim,
            name='w',
            borrow=True
        )

        self.__w = w
        self.__b = b

        y = T.scalar('y')

        # logit function.
        out = T.nnet.sigmoid( T.dot(w, x) + b )
        # cost function
        cost = -T.mean(y * T.log(out) + (1-y) * T.log( (1-out) ))
        #cost = T.mean( (y-out) ** 2 )

        
        self.__gy_w = T.grad(cost=cost, wrt=w)
        self.__gy_b = T.grad(cost=cost, wrt=b)

        updates = [
                (w, w - self.__learning_rate * self.__gy_w),
                (b, b - self.__learning_rate * self.__gy_b) 
                ]

        self.__logit = theano.function( [x], out)

        self.__training = theano.function( 
                [x, y], 
                out,
                updates = updates
                )

    def predict(self, x):
        return self.__logit(x)

    def train(self, x, y):
        self.__training(x, y)
        #print 'W: ' + str(self.__w.get_value())
        #print 'b: ' + str(self.__b.get_value())
        
def Test_SingleTrain():
    # test code.
    TestDim = 5
    EpochTimes = 100
    LearningRate = 0.2
    layer = LogisticLayer(dim = TestDim, learning_rate=LearningRate)
    single_x = numpy.random.rand(TestDim).astype(numpy.float32) - [numpy.float32(0.5)] * TestDim
    single_y = 0.
    print layer.predict(single_x)
    for i in range(EpochTimes):
        layer.train(single_x, single_y)
    print layer.predict(single_x)

if __name__=='__main__':
    Test_SingleTrain()

