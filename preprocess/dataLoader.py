import random
import pandas as pd
from PIL import Image
import numpy as np
import torch.utils.data
from torchvision.transforms import transforms
from preprocess.preprocess import process
from matplotlib import pyplot as plt
DEFAULT_W=512
DEFAULT_H=256

VAL_H=256
VAL_W=1152

info = {'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]}

class ImageDataSet(torch.utils.data.Dataset):
    # 读入映射文件
    def __init__(self,annotations_file,training:bool,tranform=None,target_transform=None):
        # super(ImageDataSet, self).__init__()
        self.img_gt=pd.read_csv(annotations_file)
        self.transform=tranform
        self.training=training
        self.target_transform=target_transform
    # 返回数据集长度

    def __len__(self):
        return self.img_gt.__len__()

    # 获取图像与groundtruth
    def __getitem__(self, index):
        img_left_path=str(self.img_gt.iloc[index,1])
        img_right_path=str(self.img_gt.iloc[index,2])
        gt_path=str(self.img_gt.iloc[index,3])
        img_left=Image.open(img_left_path)
        img_right=Image.open(img_right_path)
        gt=Image.open(gt_path)
        w,h=img_left.size
        if self.training==True:
            y=random.randint(0,h-DEFAULT_H-1)
            x=random.randint(0,w-DEFAULT_W-1)
            img_left=img_left.crop((x,y,x+DEFAULT_W,y+DEFAULT_H))

            img_left=torch.tensor(np.array(img_left)).permute(2,0,1)
            img_right=torch.tensor(np.array(img_right.crop((x,y,x+DEFAULT_W,y+DEFAULT_H)))).permute(2,0,1)

            gt=gt.crop((x,y,x+DEFAULT_W,y+DEFAULT_H))
            gt=np.ascontiguousarray(gt, dtype=np.float32) /256
        else:
            d_h=h-VAL_H
            d_w=w-VAL_W
            img_left=img_left.crop((d_w,d_h,w,h))
            img_left=torch.tensor(np.array(img_left)).permute(2,0,1)
            img_right=torch.tensor(np.array(img_right.crop((d_w,d_h,w,h)))).permute(2,0,1)
            gt=gt.crop((d_w,d_h,w,h))
            gt=np.ascontiguousarray(gt, dtype=np.float32)/256

        norm=transforms.Normalize(**(info))
        img_left=norm(img_left/255)
        img_right=norm(img_right/255)

        if self.transform:
            img_left=self.transform(img_left)
            img_right=self.transform(img_right)
        if self.target_transform:
            gt=self.target_transform(gt)
        # test
        # return img_left,img_right,gt,img_left_path,img_right_path
        # train
        return img_left,img_right,gt


if __name__=='__main__':
    dataloader=ImageDataSet('../annotation.csv','/home/jiaxi/workspace/KITTI/training')
    dataloader=torch.utils.data.DataLoader(dataloader,batch_size=1,shuffle=True,drop_last=False)
    # print(type(dataloader))

    for left in dataloader:
        print(type(left))
