# import matplotlib.pyplot as plt
# import torch
# from matplotlib import pyplot
# import math
# import numpy as np
# from einops import rearrange
# # def f(x):
# #     return np.sin(x)*(1/(abs((x+0.5)))**0.5+1/(abs((x-0.5)))**0.5)
# #
# # def g(x):
# #     x=x-1
# #     return np.sin(x)*(1/(abs((x+0.5)))**0.5+1/(abs((x-0.5)))**0.5)
# #
# # samples=np.arange(-50,51,1)
# # print(samples)
# #
# # y=f(samples)
# # print(y)
# # z=g(samples)
# # pyplot.plot(samples,y)
# # pyplot.plot(samples,z)
# # plt.show()
# d=32
# #
# # mat=np.zeros((512,512))
# def f(x):
#     mat=0
#     for k in range(d//2):
#         mat+=np.cos((x)/(10000**(2*k/d)))
#     return mat
#
# def g(x):
#     x-=3
#     mat=0
#     for k in range(d//2):
#         mat+=np.cos((x)/(10000**(2*k/d)))
#     return mat
#
# def relative_position_codeMat(C,H,W,device):
#     # desgin a relative position code matrix to reflect the Manhattan Distance
#     # It need 2 matrix to form the codeMat
#     mat_x=torch.zeros((C,W)).to(device)
#     mat_y=torch.zeros((C,H)).to(device)
#     # form the raw mat
#     for c in range(C//2):
#         for x in range(W):
#             theta=torch.tensor(x/(10000**(2*c/C)))
#             mat_x[c*2][x]=torch.sin(theta)
#             mat_x[c*2+1][x]=torch.cos(theta)
#         for y in range(H):
#             theta=torch.tensor(y/(10000**(2*c/C)))
#             mat_y[c*2][y]=torch.sin(theta)
#             mat_y[c*2+1][y]=torch.cos(theta)
#     # now the mat is [C,H,W]
#     mat_x=mat_x.expand(H,-1,-1).permute(1,0,2)
#     mat_y=mat_y.expand(W,-1,-1).permute(1,2,0)
#     mat_x=rearrange(mat_x,'c h w->(h w) c')
#     mat_y=rearrange(mat_y,'c h w->(h w) c')
#     relative_pos_code_x=torch.matmul(mat_x,torch.t(mat_x))
#     relative_pos_code_y=torch.matmul(mat_y,torch.t(mat_y))
#     return relative_pos_code_x+relative_pos_code_y
#
# mat=relative_position_codeMat(32,32,64,'cuda')
# mat=np.array(mat.cpu())
# print(mat)
#
#
#
#
#
# # samples=np.arange(-50,51,1)
# # print(samples)
# #
# # y=f(samples)
# # print(y)
# # z=g(samples)
# # pyplot.plot(samples,y-z)
# # # pyplot.plot(samples,z)
# # plt.show()
# from einops import rearrange
# import torch
# import numpy
# import torch.nn as nn
# def window_partition(embeddings):
#     '''
#     group patches by window
#     :param embeddings: shape=[N,head_num,D,patch_num]
#     :return:
#     '''
#     print('start')
#     print(embeddings)
#     N,head_num,D,patch_num=embeddings.shape
#     embeddings=embeddings.view(N,head_num,D,4,4)
#     print(embeddings)
#     print('end')
#     return rearrange(embeddings,'n h d (num_h win1) (num_w win2) -> n h (num_h num_w) d (win1 win2)',win1=2,win2=2)
#
# def window_merge(embeddings):
#     '''
#     merge patches by window
#     :param embeddings:shape=[N,head_num,window_num,D,patch_num_in_a_window]
#     :return: shape=[N,head_num,D,patch_num]
#     '''
#     N,head_num,window_num,D,_=embeddings.shape
#     embeddings=rearrange(embeddings,'n h (num_h num_w) d (win1 win2) -> n h d (num_h win1) (num_w win2)',num_h=2,win1=2)
#     return embeddings.view(N,head_num,D,-1)
#
#
#
#
#
# def features2patches(features):
#     return rearrange(features,'b c (h p1) (w p2) ->b (h w) c p1 p2',p1=2,p2=2)
#
# class flat(nn.Module):
#     def __init__(self):
#         super(flat, self).__init__()
#         self.flat=nn.Flatten(start_dim=2)
#     def forward(self,inputs):
#         return self.flat(inputs)
#
# def patches2features(patches):
#     '''
#     :param patches:shape=[N,C,P,P,patch_num]
#     :return: features: shape=[N,C,H,W]
#     '''
#     return rearrange(patches,'b c p1 p2 (h w)-> b c (h p1) (w p2)',h=4,w=4)
# h=8
# w=8
# l=[]
# for i in range(h):
#     h=list()
#     for j in range(w):
#         h.append(i*w+j)
#     l.append(h)
# print(l)
# t=torch.from_numpy(numpy.array(l))
# t=t.expand(3,8,8).unsqueeze(0)
#
# t=features2patches(t)
# f=flat()
# print(t.shape)
# print(t)
# t=f(t)
# print(t)
# print(t.shape)
# t=t.permute([0,2,1]).unsqueeze(1)  # [N,D,patch_num]
# print(t.shape)
# t=window_partition(t)
# print(t.shape)
# print(t)
# t=window_merge(t)
# print(t)
# print(t.shape)


# t=t.view(1,-1,2,2,16)
# t=patches2features(t)
# print(t)
# print(t.shape)
# import torch
# coords_h = torch.arange(4)
# coords_w = torch.arange(4)
# coords=torch.meshgrid([coords_h, coords_w])
# print(coords)
# coords = torch.stack(coords)
# print(coords)
# flaten_coords=torch.flatten(coords,start_dim=1)
# print(flaten_coords)
# coords=flaten_coords[:,:,None]-flaten_coords[:,None,:]
# coords=coords.permute(1,2,0)
# print(coords.shape)
# print()
# import logging
# Log_Format='%(asctime)s - %(levelname)s - %(message)s'
# logging.basicConfig(filename='pretrain.log',level=logging.DEBUG,format=Log_Format)
# logger=logging.getLogger("pretrain")
# logger.info('hey!')
# logger.basicConfig(filename='p.log',level=logger.DEBUG,format=Log_Format)
# logger.info('hey!')

import pandas
l_list=[]
r_list=[]
d_list=[]
def f(df):
    for i in range(len(df)):
        l=df.iloc[i,1]
        r=df.iloc[i,2]
        d=df.iloc[i,3]
        l_list.append(str(l.replace('/data/','/home/FastDataLoader/')))
        r_list.append(str(r.replace('/data/','/home/FastDataLoader/')))
        d_list.append(str(d.replace('/data/','/home/FastDataLoader/')))

train=pandas.read_csv('./train.csv')
test=pandas.read_csv('./valid.csv')
f(train)
df1=pandas.DataFrame({'left image':l_list,'right image':r_list,'disp image':d_list})
df1.to_csv('train1.csv')
l_list=[]
r_list=[]
d_list=[]
f(test)
print(l_list)
df2=pandas.DataFrame({'left image':l_list,'right image':r_list,'disp image':d_list})
df2.to_csv('valid1.csv')