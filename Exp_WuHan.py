# -*- coding:utf-8 -*-

from Utils.MyExperiment import  Exp_STNN
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(-1)
    else:
        method = sys.argv[1]

    if method=='train':
        op=Exp_STNN()
        op.train()
    elif method=='predict':
        op = Exp_STNN()
        op.predict()
    else:
        print 'keyword is wrong,please input {train,predict}'
        sys.exit(-1)

