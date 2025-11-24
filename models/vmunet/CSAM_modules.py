import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import math
import numpy as np

def custom_max(x,dim,keepdim=True): # 沿着指定的维度找到最大值
    temp_x=x
    for i in dim:
        temp_x=torch.max(temp_x,dim=i,keepdim=True)[0]  # dim=0,temp_x_0:(1,32,32,192); dim=2,temp_x_2:(1,32,1,192); dim=3,temp_x_3:(1,32,1,1);
    if not keepdim:
        temp_x=temp_x.squeeze()  # (1,32)
    return temp_x

class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super(PositionalAttentionModule,self).__init__()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(7,7),padding=3)
    def forward(self,x):  #x:(1,32,32,192)
        max_x=custom_max(x,dim=(0,1),keepdim=True)   # .squeeze(0).squeeze(0)
        avg_x=torch.mean(x,dim=(0,1),keepdim=True)   # .squeeze(0).squeeze(0)
        att=torch.cat((max_x,avg_x),dim=1) # (1,2,32,192)
        att=self.conv(att)
        att=torch.sigmoid(att)
        return x*att

class SemanticAttentionModule(nn.Module):
    def __init__(self,in_features,reduction_rate=4):
        super(SemanticAttentionModule,self).__init__()
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=in_features//reduction_rate))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=in_features//reduction_rate,out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
    def forward(self,x):
        max_x=custom_max(x,dim=(0,2,3),keepdim=False).unsqueeze(0) # 沿着指定的维度找到最大值
        avg_x=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0) # 沿着指定的维度找到平均值
        max_x=self.linear(max_x)     # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x32 and 96x6) --- 修改：SemanticAttention的参数base_num=32[vmamba_CSAM.py的681行],使得Linear的输入维度为32*2
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        att=torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)   #(1,32,1,1)
        return x*att

class SliceAttentionModule(nn.Module):
    def __init__(self,in_features,rate=4,uncertainty=True,rank=5):
        super(SliceAttentionModule,self).__init__()
        self.uncertainty=uncertainty
        self.rank=rank
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=int(in_features*rate)))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=int(in_features*rate),out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
        if uncertainty:
            self.non_linear=nn.ReLU()
            self.mean=nn.Linear(in_features=in_features,out_features=in_features)
            self.log_diag=nn.Linear(in_features=in_features,out_features=in_features)
            self.factor=nn.Linear(in_features=in_features,out_features=in_features*rank)
    def forward(self,x):  # x - (1,32,32,192)
        max_x=custom_max(x,dim=(0,2,3),keepdim=False).unsqueeze(0)  # todo: dim=(1,2,3) - max_x:(1,)
        avg_x=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0)   # todo: dim=(1,2,3) - avg_x : (1,1)
        max_x=self.linear(max_x)  # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1 and 20x80)
        # --- 修改：SliceAttention的Linear维度取决于传入的第一个参数，即class CSAM 的 num_slices参数。
        # 在vmamba_CSAM.py第691行调用CSAM()中，是第一个参数batch_size,应该更改为32，使之与传入的x维度一致

        avg_x=self.linear(avg_x)
        att=max_x+avg_x    #(1,32)
        if self.uncertainty:
            temp=self.non_linear(att)
            mean=self.mean(temp)
            diag=self.log_diag(temp).exp()
            factor=self.factor(temp)   # (1,160)
            factor=factor.view(1,-1,self.rank)   # (1,32,5)
            # 用于创建一个具有低秩形式协方差矩阵的多元正态分布,有助于在协方差矩阵接近奇异或者当协方差矩阵的秩远小于其维度时，减少存储和计算成本。
            # loc：分布的均值，形状为 batch_shape + event_shape。
            # cov_factor：协方差矩阵低秩形式的因子部分，形状为 batch_shape + event_shape + (rank,)。
            # cov_diag：协方差矩阵的低秩形式的对角部分，形状为 batch_shape + event_shape。
            dist=td.LowRankMultivariateNormal(loc=mean,cov_factor=factor,cov_diag=diag)   # LowRankMultivariateNormal(loc: torch.size([1, 32]), cov factor: torch.size([1, 32, 5]), cov diag: torch.size([1, 32))
            att=dist.sample()   # att:(1,32) - 在低秩形式协方差矩阵上进行采样，生成一个符合该分布特性的样本张量 attention
        att=torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        # if x.shape[0] != 1:   # todo: add
        #     alpha = x.shape[0]//att.shape[0]
        #     att = att.repeat(alpha,1,1,1)
        return x*att   # x:(1,32,32,192),att:(32,1,1,1);   x:(32,16,16,384),att:(16,1,1,1)


class CSAM(nn.Module):
    def __init__(self,num_slices,num_channels,semantic=True,positional=True,slice=True,uncertainty=True,rank=5):
        super(CSAM,self).__init__()
        self.semantic=semantic
        self.positional=positional
        self.slice=slice
        if semantic:
            self.semantic_att=SemanticAttentionModule(num_channels)
        if positional:
            self.positional_att=PositionalAttentionModule()
        if slice:
            self.slice_att=SliceAttentionModule(num_slices,uncertainty=uncertainty,rank=rank)
    def forward(self,x):  #x - (1,32,32,192)
        if self.semantic:
            x=self.semantic_att(x)  #x - (1,32,32,192)
        if self.positional:
            x=self.positional_att(x)  #x - (1,32,32,192)
        if self.slice:
            x=self.slice_att(x)  #?? x - (32,32,32,192)
        return x

