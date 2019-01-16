# coding: utf-8
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from my_class import MyDataset, Net, DropoutFC, VGGNet


net = DropoutFC()     # 创建一个网络
pretrained_dict = torch.load('./Result/01-14_11-51-20/net_params.pkl')
net.load_state_dict(pretrained_dict)

writer = SummaryWriter(log_dir='./Result/visual_weights')
params = net.state_dict()
for k, v in params.items():
    if 'conv' in k and 'weight' in k:

        c_int = v.size()[1]     # 输入层通道数
        c_out = v.size()[0]     # 输出层通道数

        # 以feature map为单位，绘制一组卷积核，一张feature map对应的卷积核个数为输入通道数
        for j in range(c_out):
            print(k, v.size(), j)
            kernel_j = v[j, :, :, :].unsqueeze(1)       # 压缩维度，为make_grid制作输入
            kernel_grid = vutils.make_grid(kernel_j, normalize=True, scale_each=True, nrow=c_int)   # 1*输入通道数, w, h
            writer.add_image(k+'_split_in_channel', kernel_grid, global_step=j)     # j 表示feature map数

        # 将一个卷积层的卷积核绘制在一起，每一行是一个feature map的卷积核
        k_w, k_h = v.size()[-1], v.size()[-2]
        kernel_all = v.view(-1, 1, k_w, k_h)
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=c_int)  # 1*输入通道数, w, h
        writer.add_image(k + '_all', kernel_grid, global_step=666)
writer.close()