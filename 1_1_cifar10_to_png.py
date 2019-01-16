# coding:utf-8
"""
    将cifar10的data_batch_12345 转换成 png格式的图片
    每个类别单独存放在一个文件夹，文件夹名称为0-9
"""
from scipy.misc import imsave
import numpy as np
import os
import pickle

data_dir = './Data/cifar-10-batches-py/'
train_o_dir = './Data/cifar-10-png/raw_train/'
test_o_dir = './Data/cifar-10-png/raw_test/'

Train = True  # 不解压训练集，仅解压测试集


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict_ = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict_


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


# 生成训练集图片，
if __name__ == '__main__':
    if Train:
        for j in range(1, 6):
            data_path = data_dir + "data_batch_" + str(j)  # data_batch_12345
            train_data = unpickle(data_path)
            print(data_path + " is loading...")

            for i in range(0, 10000):
                img = np.reshape(train_data[b'data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)

                label_num = str(train_data[b'labels'][i])
                o_dir = os.path.join(train_o_dir, label_num)
                my_mkdir(o_dir)

                img_name = label_num + '_' + str(i + (j - 1) * 10000) + '.png'
                img_path = os.path.join(o_dir, img_name)
                imsave(img_path, img)
            print(data_path + " loaded.")

    print("test_batch is loading...")

    # 生成测试集图片
    test_data_path = data_dir + "test_batch"
    test_data = unpickle(test_data_path)
    for i in range(0, 10000):
        img = np.reshape(test_data[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)

        label_num = str(test_data[b'labels'][i])
        o_dir = os.path.join(test_o_dir, label_num)
        my_mkdir(o_dir)

        img_name = label_num + '_' + str(i) + '.png'
        img_path = os.path.join(o_dir, img_name)
        imsave(img_path, img)

    print("test_batch loaded.")
