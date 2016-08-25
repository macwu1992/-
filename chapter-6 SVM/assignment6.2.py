# coding: utf-8

import os
import sys

sys.path.append('/Users/Tong/workspace/MachineLearning-BOOK_ZhouZhiHua/libsvm/python')

import svmutil as svm

y, x = svm.svm_read_problem('../西瓜数据集3.0a.txt')

linear_svm = svm.svm_train(y, x, '-t 0')
RBF_svm = svm.svm_train(y, x, '-t 2')

linear_svm_file = 'linear_svm.model'
RBF_svm_file = 'RBF_svm.model'

svm.svm_save_model(linear_svm_file, linear_svm)
svm.svm_save_model(RBF_svm_file, RBF_svm)