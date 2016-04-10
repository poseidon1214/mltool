#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import sys

import numpy
import sklearn

import pydev

class NameIDTransformer:
    def __init__(self):
        self.__mapping = {}
        self.__names = []

    def read(self, name):
        if name not in self.__mapping:
            self.__mapping[name] = len(self.__names)
            self.__names.append(name)
        return self.__mapping[name]

    def id(self, name):
        return self.__mapping.get(name, None)

    def name(self, id):
        return self.__names[id]

    def debug(self):
        print >> sys.stderr, self.__names

class DataReader(object):
    def __init__(self):
        self.__seperator = ','
        self.__expect_column_count = 5
        self.__target_column = 4
        self.__name_to_id_dict = {}
        self.__ignore_first_row = False
        self.__ignore_columns = set()
        self.__target_trans = NameIDTransformer()

        self.__X = []
        self.__Y = []

    def config(self, config_name):
        if '#' not in config_name:
            raise Exception('bad config format: [ config#secion ] is needed.')

        config_file, section = config_name.split('#')
        config = pydev.VarConfig()
        config.read(config_file)

        self.__seperator = config.get(section, 'seperator', default=',')
        self.__expect_column_count = int( config.get(section, 'fix_columns', default='-1') )
        self.__target_column = int( config.get(section, 'target_column', default='0') ) - 1
        self.__ignore_first_row = int( config.get(section, 'ignore_first_row', default='0') )

        s = config.get(section, 'name_to_id', default='')
        if s:
            self.__name_to_id_dict = dict( map(lambda x:(int(x)-1, NameIDTransformer()), s.split(',')) )

        s = config.get(section, 'ignore_columns', default='')
        if s:
            self.__ignore_columns = set( map(lambda x:int(x)-1, s.split(',')) )

        if self.__expect_column_count >= 0 and self.__target_column < 0:
            self.__target_column = self.__expect_column_count - 1

        print >> sys.stderr, 'Load config over'
        print >> sys.stderr, 'seperator : [%s]' % self.__seperator
        print >> sys.stderr, 'columns : %d (-1 : indicates from first useable row)' % self.__expect_column_count
        print >> sys.stderr, 'target  : %d (0 based, -1 indicates last column)' % self.__target_column
        print >> sys.stderr, 'name_to_id : %s' % (','.join(map(str, self.__name_to_id_dict.keys())))
        print >> sys.stderr, 'ignore_columns : %s' % (','.join(map(str, self.__ignore_columns)))
        print >> sys.stderr, 'ignore_first_row : %d' % self.__ignore_first_row

    def read(self, filename):
        self.__X = []
        self.__Y = []
        first_row = True
        for row in pydev.foreach_row(
                file(filename), 
                seperator=self.__seperator):

            if first_row and self.__ignore_first_row:
                first_row = False
                continue

            if self.__expect_column_count < 0:
                self.__expect_column_count = len(row)
                if self.__target_column < 0:
                    self.__target_column = self.__expect_column_count - 1
                print >> sys.stderr, 'columns set to %d, target:%d' % (self.__expect_column_count, self.__target_column)
            elif len(row) != self.__expect_column_count:
                    continue

            row = map(lambda x:x.strip(), row)

            for cid, trans in self.__name_to_id_dict.iteritems():
                row[cid] = trans.read(row[cid])

            row[self.__target_column] = self.__target_trans.read( row[self.__target_column] )
            y = row[self.__target_column]

            filter_row = map(
                            lambda (rid, value): float(value),
                            filter(
                                lambda (rid, value):rid not in self.__ignore_columns and rid!=self.__target_column, 
                                enumerate(row))
                            )

            x = numpy.array( filter_row )
            x = x.astype(numpy.float32)
            

            self.__X.append(x)
            self.__Y.append(y)

        #self.__target_trans.debug()
        print >> sys.stderr, 'Data load (%d records)' % len(self.__X)

    @property
    def data(self):
        return self.__X

    @property
    def target(self):
        return self.__Y

if __name__=='__main__':
    arg = pydev.Arg('A simple tool for machine learning based ont scikit-learn')
    arg.str_opt('filename', 'f', 'input file of data')
    arg.str_opt('test', 't', 'data file for test')
    arg.str_opt('reader_config', 'r', 'config of reader, format: filename#section')
    opt = arg.init_arg()

    reader = DataReader()
    if opt.reader_config:
        reader.config(opt.reader_config)
    reader.read(opt.filename)

    train_X = reader.data
    train_Y = reader.target

    if opt.test:
        reader.read(opt.test)
        test_X = reader.data
        test_Y = reader.target
    
    else:
        test_X = train_X
        test_Y = train_Y

    from sklearn import svm
    from sklearn import linear_model
    from sklearn import neighbors
    from sklearn import naive_bayes
    from sklearn import tree

    models = (
        ('svm', svm.SVC() ),
        ('lr', linear_model.LogisticRegression() ),
        ('knn', neighbors.KNeighborsClassifier() ),
        ('gnb', naive_bayes.GaussianNB() ),
        ('tree', tree.DecisionTreeClassifier() )
        )

    def report(pred, target):
        #print pred[:100]
        #print target[:100]
        diff_count = len(filter(lambda x:x!=0, pred - target))
        precision = (len(target) - diff_count) * 100. / len(target)
        print 'Precision: %.2f%% (%d/%d)' % (precision, len(target)-diff_count, len(target))

    for name, model in models:
        print >> sys.stderr, '====> model [%s] <====' % name
        model.fit(train_X, train_Y)
        print >> sys.stderr, 'Training over.'
        pred = model.predict(test_X)
        print >> sys.stderr, 'Predict over.'
        report(pred, test_Y)
        
