import torch
from torch import nn 
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
#from timm.models.vision_transformer import VisionTransformer, _cfg
#from timm.models.registry import register_model
#from timm.models.layers import trunc_normal_
from functools import partial
#from models_new import Encoder
#from se_module import SELayer

class ResNetBlock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)
        if not self.first :
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp =  self.conv11(x)
        if not self.first:
            out = self.pre_bn(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(x)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp
        return out

class SEResNetBlock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(SEResNetBlock, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)
        self.seblock = SELayer(channel=32)
        if not self.first :
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp =  self.conv11(x)
        if not self.first:
            out = self.pre_bn(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(x)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp
        out = self.seblock(out)
        return out

class MFCCModel(nn.Module):
    def __init__(self):
        super(MFCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32,  False)
        self.block3 = ResNetBlock(32, 32,  False)
        self.block4= ResNetBlock(32, 32, False)
        self.block5= ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        print(x.shape)
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        global feature
        feature = out
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class SpectrogramModel(nn.Module):
    def __init__(self):
        super(SpectrogramModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32,  False)
        self.block3 = ResNetBlock(32, 32,  False)
        self.block4= ResNetBlock(32, 32, False)
        self.block5= ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.block10 = ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        print(x.shape)
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        #out = self.block2(out)
        #out = self.mp(out)
        out = self.block3(out)
        #out = self.block4(out)
        #out = self.mp(out)
        out = self.block5(out)
        #out = self.block6(out)
        #out = self.mp(out)
        out = self.block7(out)
        #out = self.block8(out)
        #out = self.mp(out)
        out = self.block9(out)
        #out = self.block10(out)
        #out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        #out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        global feature
        feature = out
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class CQCCModel(nn.Module):
    def __init__(self):
        super(CQCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32, False)
        self.block3 = ResNetBlock(32, 32, False)
        self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.block10 = ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        print(x.shape)
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        global feature
        feature = out
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class SEMFCCModel(nn.Module):
    def __init__(self):
        super(SEMFCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = SEResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = SEResNetBlock(32, 32, False)
        self.block3 = SEResNetBlock(32, 32, False)
        self.block4 = SEResNetBlock(32, 32, False)
        self.block5 = SEResNetBlock(32, 32, False)
        self.block6 = SEResNetBlock(32, 32, False)
        self.block7 = SEResNetBlock(32, 32, False)
        self.block8 = SEResNetBlock(32, 32, False)
        self.block9 = SEResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        print(x.shape)
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        global feature
        feature = out
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class SESpectrogramModel(nn.Module):
    def __init__(self):
        super(SESpectrogramModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = SEResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = SEResNetBlock(32, 32, False)
        self.block3 = SEResNetBlock(32, 32, False)
        self.block4 = SEResNetBlock(32, 32, False)
        self.block5 = SEResNetBlock(32, 32, False)
        self.block6 = SEResNetBlock(32, 32, False)
        self.block7 = SEResNetBlock(32, 32, False)
        self.block8 = SEResNetBlock(32, 32, False)
        self.block9 = SEResNetBlock(32, 32, False)
        self.block10 = SEResNetBlock(32, 32, False)
        self.block11 = SEResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        print(x.shape)
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        # out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        # out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        # out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        # out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        # out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        global feature
        feature = out
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class SECQCCModel(nn.Module):
    def __init__(self):
        super(SECQCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = SEResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = SEResNetBlock(32, 32, False)
        self.block3 = SEResNetBlock(32, 32, False)
        self.block4 = SEResNetBlock(32, 32, False)
        self.block5 = SEResNetBlock(32, 32, False)
        self.block6 = SEResNetBlock(32, 32, False)
        self.block7 = SEResNetBlock(32, 32, False)
        self.block8 = SEResNetBlock(32, 32, False)
        self.block9 = SEResNetBlock(32, 32, False)
        self.block10 = SEResNetBlock(32, 32, False)
        self.block11 = SEResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        print(x.shape)
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        global feature
        feature = out
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class RESModel(nn.Module):
    def __init__(self):
        super(RESModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32, False)
        self.block3 = ResNetBlock(32, 32, False)
        self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.block10 = ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer

feature = 0

class SEDenseNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=2, growth_rate=32, block_layers=[3,3,3,2]):
        super(SEDenseNet, self).__init__()
        # 模型开始部分的卷积池化层
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            SELayer(channel=32)
            )

        channels = 32
        block = []
        # 循环添加dense_block模块，并在非最后一层的层末尾添加过渡层
        for i, layers in enumerate(block_layers):
            block.append(SELayer(channel=channels))
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(SELayer(channel=channels))
                # 每经过一个dense_block模块，则在后面添加一个过渡模块，通道数减半channels//2
                block.append(transition(channels, channels // 2))
                channels = channels // 2
        self.block2 = nn.Sequential(*block) #将block层展开赋值给block2
        # 添加其他最后层
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('selayer0b', SELayer(channel=channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        self.fc1 = nn.Linear(8664, 128)
        self.lrelu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(128, num_classes)
        #self.logsoftmax = nn.LogSoftmax(dim=1)
        self.sp = nn.Softplus()
        self.W = nn.Parameter(torch.randn(2, 2))
    def forward(self, x, y, global_step):
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.block2(x)
        #print(x.shape)#[8, 152, 2, 16]

        x = x.view(x.shape[0], -1)#[8,]
        #print(x.shape)
        x = self.fc1(x)
        global feature
        feature = x
        #print(x.shape)#[8, ]
        x = self.lrelu(x)
        x = self.fc2(x)#[8,2]
        #print(x.shape)

        #softmax
        evidence = self.sp(x)
        loss = 0
        alpha = dict()
        
        alpha = evidence + 1
        loss += ce_loss(y, alpha, 2, global_step, 50)
        loss = torch.mean(loss) 
        return evidence, loss

        #A-softmax
        # _W = F.normalize(self.W, dim=0)  # 得到W/W模，而norm(W)是的得到W的模
        # _X = F.normalize(x, dim=1)
        # cosine = torch.matmul(_X, _W) / 10
        # s = 1
        # a = torch.acos(cosine * 0.999)
        # top = torch.exp(s * torch.cos(a + 1) * 10)
        # _top = torch.exp(s * cosine * 10)
        # bottom = torch.sum(torch.exp(s * cosine * 10), dim=1, keepdim=True)
        # return top / (bottom - _top + top)


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label.cuda() * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E.cuda() * (1 - label.cuda()) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

class seDenseTransNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=2, growth_rate=32, block_layers=[3,3,3,2]):
        super(seDenseTransNet, self).__init__()
        # 模型开始部分的卷积池化层
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            SELayer(channel=32)
            )

        channels = 32
        block = []
        # 循环添加dense_block模块，并在非最后一层的层末尾添加过渡层
        for i, layers in enumerate(block_layers):
            block.append(SELayer(channel=channels))
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(SELayer(channel=channels))
                # 每经过一个dense_block模块，则在后面添加一个过渡模块，通道数减半channels//2
                block.append(transition(channels, channels // 2))
                channels = channels // 2
        self.block2 = nn.Sequential(*block) #将block层展开赋值给block2
        # 添加其他最后层
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('selayer0b', SELayer(channel=channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        self.encoder = Encoder(32, 6, 8, 512, 1024)
        self.fc1 = nn.Linear(32, 512)
        self.lrelu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(77824, 128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.block2(x)
        #print(x.shape)#[8, 152, 2, 16]

        x = x.view(x.shape[0], x.shape[1], - 1)
        #print(x.shape)
        #x = self.fc1(x)
        #print(x.shape)#[8, 152, 512]
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)#[8, ]
        global feature
        feature = x
        x = self.lrelu(x)
        x = self.fc2(x)#[8,2]
        #print(x.shape)
        x = self.logsoftmax(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=2, growth_rate=32, block_layers=[3,3,3,2]):
        super(DenseNet, self).__init__()
        # 模型开始部分的卷积池化层
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #SELayer(channel=32)
            )

        channels = 32
        block = []
        # 循环添加dense_block模块，并在非最后一层的层末尾添加过渡层
        for i, layers in enumerate(block_layers):
            #block.append()
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                # 每经过一个dense_block模块，则在后面添加一个过渡模块，通道数减半channels//2
                block.append(transition(channels, channels // 2))
                channels = channels // 2
        self.block2 = nn.Sequential(*block) #将block层展开赋值给block2
        # 添加其他最后层
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        self.fc1 = nn.Linear(2280, 128)
        self.lrelu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(128, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.block2(x)
        #print(x.shape)#[8, 152, 2, 16]

        x = x.view(x.shape[0], -1)#[8,]
        #print(x.shape)
        x = self.fc1(x)
        global feature
        feature = x
        #print(x.shape)#[8, ]
        x = self.lrelu(x)
        x = self.fc2(x)#[8,2]
        #print(x.shape)

        x = self.logsoftmax(x)
        return x

'''
class DenseNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=2, growth_rate=32, block_layers=[3, 3, 3, 2]):
        super(DenseNet, self).__init__()
        # 模型开始部分的卷积池化层
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        channels = 32
        block = []
        # 循环添加dense_block模块，并在非最后一层的层末尾添加过渡层
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                # 每经过一个dense_block模块，则在后面添加一个过渡模块，通道数减半channels//2
                block.append(transition(channels, channels // 2))
                channels = channels // 2
        self.block2 = nn.Sequential(*block)  # 将block层展开赋值给block2
        # 添加其他最后层
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))

        for p in self.parameters():
            p.requires_grad=False

        self.fc1 = nn.Linear(77824, 128)
        self.lrelu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(128, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.encoder = Encoder(32, 6, 8, 512, 1024)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.block2(x)
        #print(x.shape)  # [8, 152, 2, 16]

        x = x.view(x.shape[0], x.shape[1], - 1)  # [8,152,32]
        #print(x.shape)
        #x = self.fc1(x)  # [8,152,512]
        x = self.encoder(x)   # [8,152,512]
        #print("Encoder output", x.shape)  # [8, 152, 512]
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)  # [8,2]
        #print(x.shape)
        global feature
        feature = x
        x = self.logsoftmax(x)
        return x
'''
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc2(x)
        out = self.logsoftmax(out)
        return out

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=True, resolution=[6, 6]):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CQCCBotNet(nn.Module):
    def __init__(self):
        super(CQCCBotNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Bottleneck(32, 32)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        #self.block2 = Bottleneck(32, 32)
        #self.block3 = Bottleneck(32, 32)
        #self.block4 = Bottleneck(32, 32)
        #self.block5 = Bottleneck(32, 32)
        #self.block6 = Bottleneck(32, 32)
        #self.block7 = Bottleneck(32, 32)
        #self.block8 = Bottleneck(32, 32)
        #self.block9 = Bottleneck(32, 32)
        #self.block10 = Bottleneck(32, 32)
        #self.block11 = Bottleneck(32, 32)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        #out = self.block3(out)
        # out = self.block4(out)
        #out = self.mp(out)
        #out = self.block5(out)
        # out = self.block6(out)
        #out = self.mp(out)
        #out = self.block7(out)
        # out = self.block8(out)
        #out = self.mp(out)
        #out = self.block9(out)
        # out = self.block10(out)
        #out = self.mp(out)
        #out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out

class LFCCBotNet(nn.Module):
    def __init__(self):
        super(LFCCBotNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Bottleneck(32, 32)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = Bottleneck(32, 32)
        self.block3 = Bottleneck(32, 32)
        self.block4 = Bottleneck(32, 32)
        self.block5 = Bottleneck(32, 32)
        self.block6 = Bottleneck(32, 32)
        self.block7 = Bottleneck(32, 32)
        self.block8 = Bottleneck(32, 32)
        self.block9 = Bottleneck(32, 32)
        self.block10 = Bottleneck(32, 32)
        self.block11 = Bottleneck(32, 32)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class SpectBotNet(nn.Module):
    def __init__(self):
        super(SpectBotNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Bottleneck(32, 32)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = Bottleneck(32, 32)
        self.block3 = Bottleneck(32, 32)
        self.block4 = Bottleneck(32, 32)
        self.block5 = Bottleneck(32, 32)
        self.block6 = Bottleneck(32, 32)
        self.block7 = Bottleneck(32, 32)
        self.block8 = Bottleneck(32, 32)
        self.block9 = Bottleneck(32, 32)
        self.block10 = Bottleneck(32, 32)
        self.block11 = Bottleneck(32, 32)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        # out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        # out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        # out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        # out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        # out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class MFCCBotNet(nn.Module):
    def __init__(self):
        super(MFCCBotNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = Bottleneck(32, 32)
        self.block3 = Bottleneck(32, 32)
        self.block4 = Bottleneck(32, 32)
        self.block5 = Bottleneck(32, 32)
        self.block6 = Bottleneck(32, 32)
        self.block7 = Bottleneck(32, 32)
        self.block8 = Bottleneck(32, 32)
        self.block9 = Bottleneck(32, 32)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        x = x.reshape((32, 1, 60, 399))
        outputs = 0
        i = 0
        for name, smodule in self.submodule._modules.items():
            for name, module in smodule._modules.items():
                i = i + 1
                print("i is {}".format(i))
                x = module(x)
                if name == "block2":
                    x = x.reshape((32, 4864))
                if name == self.extracted_layers:
                    outputs = x
                    print("output is {}".format(outputs))
                    return outputs
        return outputs

def getFeature():
    return feature



