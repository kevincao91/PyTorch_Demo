# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from my_class import MyDataset, Net, DropoutFC, validate_cuda, show_confMat
from tensorboardX import SummaryWriter
from datetime import datetime

train_txt_path = './Data/train.txt'
valid_txt_path = './Data/valid.txt'
test_txt_path = './Data/test.txt'
mean_data_path = './Data/mean_data.txt'

classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_bs = 16
valid_bs = 16
test_bs = 16
lr_init = 0.001
max_epoch = 50

# log
result_dir = './Result/'

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

print(torch.cuda.is_available())
# 返回True代表支持，False代表不支持

# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
with open(mean_data_path, 'r') as f:
    lines = f.readlines()

normMean = [float(i) for i in lines[0].split()]
normStd = [float(i) for i in lines[1].split()]
# normMean = [0.49387893, 0.4849765, 0.4503186]
# normStd = [0.24682394, 0.24297716, 0.26187313]

normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

testTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)
test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)
test_loader = DataLoader(dataset=test_data, batch_size=test_bs)


# ------------------------------------ step 2/5 : 定义网络------------------------------------

net = DropoutFC().cuda()  # 创建一个网络
net.initialize_weights()  # 初始化权值

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss().cuda()  # 选择损失函数
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)  # 选择优化器
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.50)  # 设置学习率下降策略
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       factor=0.1,
                                                       patience=10,
                                                       verbose=True,
                                                       min_lr=0, eps=1e-08)  # 设置学习率下降策略

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------

for epoch in range(max_epoch):

    net.train()  # 训练模式
    acc_avg_epoch = 0.0  # 记录一个epoch的acc均值
    acc_sigma = 0.0  # 记录n个Iteration的acc之和
    acc_sigma_epoch = 0.0  # 记录一个epoch的acc之和
    loss_avg_epoch = 0.0  # 记录一个epoch的loss均值
    loss_sigma = 0.0  # 记录n个Iteration的loss之和
    loss_sigma_epoch = 0.0  # 记录一个epoch的loss之和
    scheduler.step(acc_avg_epoch)  # 更新学习率

    for i, data in enumerate(train_loader):

        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        # 正确率
        total = labels.size(0)
        correct = (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
        acc = correct / total
        acc_sigma += acc
        acc_sigma_epoch += acc
        # 损失计数
        loss_sigma += loss.item()
        loss_sigma_epoch += loss.item()

        # 每50个iteration 打印一次训练信息，loss为50个iteration的平均，acc为50个iteration的平均
        n_num = 50
        if i % n_num == n_num-1:
            loss_avg = loss_sigma / n_num
            loss_sigma = 0.0
            acc_avg = acc_sigma / n_num
            acc_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, acc_avg))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

    # 每个epoch的 Loss, accuracy, learning rate
    lr_now = [group['lr'] for group in optimizer.param_groups][0]
    acc_avg_epoch = acc_sigma_epoch / len(train_loader)
    loss_avg_epoch = loss_sigma_epoch / len(train_loader)
    print("Training: Epoch[{:0>3}/{:0>3}] Loss_Avg_Epoch: {:.4f} Acc_Avg_Epoch:{:.2%} Lr: {:.8f}".format(
        epoch + 1, max_epoch, loss_avg_epoch, acc_avg_epoch, lr_now))
    # 记录Loss, accuracy, learning rate
    writer.add_scalar('learning rate', lr_now, epoch)
    writer.add_scalars('Loss_group', {'train_loss_avg_epoch': loss_avg_epoch}, epoch)
    writer.add_scalars('Accuracy_group', {'train_acc_avg_epoch': acc_avg_epoch}, epoch)

    # 每个epoch，记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------

    # 每5次测试一次
    if epoch % 5 == 4:

        net.eval()    # 测试模式
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵

        for i, data in enumerate(valid_loader):

            # 获取图片和标签
            images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()

            # forward
            outputs = net(images)
            outputs.detach_()

            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor
            labels = labels.cpu()
            predicted = predicted.cpu()

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].numpy()
                pre_i = predicted[j].numpy()
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Loss:{:.4f} Accuracy:{:.2%}'.format('Valid',
                                                          loss_sigma / len(valid_loader),
                                                          conf_mat.trace() / conf_mat.sum()))  # trace()方阵的迹
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)

print('Finished Training')

# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
net_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(net.state_dict(), net_save_path)

conf_mat_train, train_acc = validate_cuda(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate_cuda(net, valid_loader, 'valid', classes_name)
conf_mat_test, test_acc = validate_cuda(net, test_loader, 'test', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
show_confMat(conf_mat_test, classes_name, 'test', log_dir)

with open(os.path.join(log_dir, 'acc_data.txt'), 'w') as f:
    f.write('train_acc=' + str(train_acc) + '\n')
    f.write('valid_acc=' + str(valid_acc) + '\n')
    f.write('test_acc=' + str(test_acc) + '\n')
