# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集
"""

import os
import glob
import random
import shutil

dataset_train_dir = './Data/cifar-10-png/raw_train/'
dataset_test_dir = './Data/cifar-10-png/raw_test/'
train_dir = './Data/MaxData/train/'
valid_dir = './Data/MaxData/valid/'
test_dir = './Data/MaxData/test/'

train_per = 0.9
valid_per = 0.1


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    # train & valid
    for root, dirs, files in os.walk(dataset_train_dir):
        for sDir in dirs:
            imgs_list = glob.glob(os.path.join(root, sDir)+'/*.png')
            random.seed(666)
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)

            train_point = int(imgs_num * train_per)

            out_dir = train_dir + sDir + '/'
            makedir(out_dir)
            out_dir = valid_dir + sDir + '/'
            makedir(out_dir)

            for i in range(imgs_num):
                if i < train_point:
                    out_dir = train_dir + sDir + '/'
                else:
                    out_dir = valid_dir + sDir + '/'

                out_path = out_dir + os.path.split(imgs_list[i])[-1]
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, train:{}, valid:{}'.format(sDir, train_point, imgs_num-train_point))

    # test
    for root, dirs, files in os.walk(dataset_test_dir):
        for sDir in dirs:
            imgs_list = glob.glob(os.path.join(root, sDir)+'/*.png')
            imgs_num = len(imgs_list)

            out_dir = test_dir + sDir + '/'
            makedir(out_dir)

            for i in range(imgs_num):
                out_path = out_dir + os.path.split(imgs_list[i])[-1]
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, test:{}'.format(sDir, imgs_num))
