# -*- coding: utf-8 -*-
import numpy as np
from logRegressTest import *


'''加载数据'''
def loadFile():
    train_x =[]
    train_y =[]
    fileIn = open('J:/mission/data.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0,float(lineArr[1]), float(lineArr[2]), float(lineArr[2])])#y=w0+x1*w1+x2*w2,(3个维度)
        train_y.append(float(lineArr[0]))
    return np.mat(train_x), np.mat(train_y).transpose()



'''逻辑回归测试'''
def logRegresMain():
    print("step 1: loading data...")
    train_x, train_y = loadFile()
    test_x = train_x; test_y = train_y

    print("step 2: training...")
    alpha = 0.01
    maxIter = 200

    #gradDescent ,stocGradDescent ,smoothStocGradDescent
    optimizeType = 'gradDescent'#调用的方法

    opts = {'alpha': alpha, 'maxIter': maxIter, 'optimizeType': optimizeType}
    optimalWeights = train_logRegres(train_x, train_y, opts)

    ## step 3: testing
    print("step 3: testing...")
    accuracy = test_LogRegres(optimalWeights, test_x, test_y)

    ## step 4: show the result
    print("step 4: show the result...")
    print('The classify accuracy is: %.3f%%' % (accuracy * 100))
    showLogRegres(optimalWeights, train_x, train_y)


if __name__=='__main__':
    logRegresMain()
