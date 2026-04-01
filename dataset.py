import os
import cv2
import torch
from torch.utils.data import Dataset

class TomatoDataset(Dataset):

    def __init__(self,img_dir,label_dir):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.images=os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):

        img_name=self.images[idx]

        img_path=os.path.join(self.img_dir,img_name)

        label_path=os.path.join(
            self.label_dir,
            img_name.replace(".jpg",".txt").replace(".png",".txt")
        )

        image=cv2.imread(img_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        boxes=[]
        labels=[]

        if os.path.exists(label_path):

            with open(label_path) as f:

                for line in f.readlines():

                    c,xmin,ymin,xmax,ymax=map(int,line.split())

                    boxes.append([xmin,ymin,xmax,ymax])
                    labels.append(c)

        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.int64)

        target={}
        target["boxes"]=boxes
        target["labels"]=labels

        image=torch.tensor(image).permute(2,0,1).float()/255

        return image,target