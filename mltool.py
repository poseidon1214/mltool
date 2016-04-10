#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import sklearn
import numpy

class DataReader(object):
    def __init__(self):
        pass

    def read(self, filename):
        pass

    @property
    def data(self):
        pass

    @property
    def target(self):
        pass

if __name__=='__main__':
    reader = DataReader()
    reader.read(sys.argv[1])

    models = []

    def report(pred, target):
        diff_count = len(filter(lambda x:x!=0, pred - target))
        precision = diff_count / len(target)
        print 'Precision: %.2f%% (%d/%d)' % (precision, diff_count, len(target))

    for model in models:
        model.fit(reader.data, reader.target)
        pred = model.predict(reader.data)

        report(pred, target)
        
