import random

import numpy
import torch
import torch.utils.data as data
import numpy as np
import pandas
from preprocess import preprocess
from deepPruner.config import config
from PIL import Image
import preprocess.pfm as pfm
DEFUALT_W=config.defualt_w
DEFUALT_H=config.defualt_h

class SceneflowDataLoader(data.Dataset):
    def __init__(self,file_path,train:bool):
        self.files=pandas.read_csv(file_path)
        self.is_training=train

    def __getitem__(self, index):
        # packed=np.load(self.files.iloc[index,1]).transpose((2,0,1))
        # packed=torch.tensor(packed).permute(2,0,1)
        left_path=self.files.iloc[index,1]
        right_path=self.files.iloc[index,2]
        dip_path=self.files.iloc[index,3]
        left_img=Image.open(left_path)
        right_img=Image.open(right_path)
        left=np.array(left_img).transpose((2,0,1)).astype(np.float32)
        right=np.array(right_img).transpose((2,0,1)).astype(np.float32)
        disp,scale=pfm.read_pfm(dip_path)

        if self.is_training:
            h,w=left.shape[1:3]
            dh,dw=DEFUALT_H,DEFUALT_W
            h0=random.randint(0,h-dh)
            w0=random.randint(0,w-dw)
            # crop
            left=left[:,h0:h0+dh,w0:w0+dw]
            right=right[:,h0:h0+dh,w0:w0+dw]
            disp=disp[h0:h0+dh,w0:w0+dw]

        else:
            scale=config.cost_aggregator_scale*8
            h,w=left.shape[1:3]
            ch=(h//scale)*scale
            cw=(w//scale)*scale
            dh=h-ch
            dw=w-cw+64
            left=left[:,dh:,dw:]
            right=right[:,dh:,dw:]
            disp=disp[dh:,dw:]
        # left=torch.tensor(left)
        # right=torch.tensor(right)
        disp=torch.tensor(disp)
        # print(type(left[0][0][0]))
        left=preprocess.process()(left.transpose([1,2,0]))
        right=preprocess.process()(right.transpose([1,2,0]))
        disp=disp.unsqueeze(0)
        # 训练时

        # return left,right,disp
        # 测试可视化时
        left_img=numpy.array(left_img)
        right_img=numpy.array(right_img)
        return left,right,disp,left_img,right_img

    def __len__(self):
        return self.files.__len__()

if __name__=='__main__':
    disp=pfm.read_pfm('/data/sceneflow/monkaa/disparity/eating_camera2_x2/left/0102.pfm')
    print(disp.shape)