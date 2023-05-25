import os
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transform
from PIL import Image

data_set_path=os.path.join(os.getcwd(),"data")

train_x_data_path=os.path.join(data_set_path,"trainA")
train_y_data_path=os.path.join(data_set_path,"trainB")
test_x_data_path=os.path.join(data_set_path,"testA")
test_y_data_path=os.path.join(data_set_path,"testB")


#constants

batch_size=1

class FacadesDataSet(Dataset):
    def __init__(self,train_x_path,train_y_path,transforms=None,target_transforms=None):
        self.x_path=train_x_path
        self.y_path=train_y_path
        self.transforms=transforms
        self.targettransforms=target_transforms
        self.xindex_list=dict(
            zip(
                list(range(len(os.listdir(self.x_path)))),
                list(sorted(os.listdir(self.x_path)))
        )
        )
        self.y_index_list=dict(

             zip(
                list(range(len(os.listdir(self.y_path)))),
                list(sorted(os.listdir(self.y_path)))
        )

        )

    
    def __len__(self):
        return len(os.listdir(self.x_path))

    def __getitem__(self,idx):
        img_a=Image.open(os.path.join(self.x_path,self.xindex_list[idx]))
        img_b=Image.open(os.path.join(self.y_path,self.y_index_list[idx]))
        if self.transforms is not None:
            img_a=self.transforms(img_a)
        if self.targettransforms is not None:
            img_b=self.transforms(img_b)

        return img_a,img_b


c_transforms=transform.Compose([transform.ToTensor()])

train_dataset=FacadesDataSet(train_x_data_path,train_y_data_path,transforms=c_transforms,target_transforms=c_transforms)

test_dataset=FacadesDataSet(test_x_data_path,test_y_data_path,transforms=c_transforms,target_transforms=c_transforms)

train_data_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_data_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


def get_train_data_loader():
    return train_data_loader

def get_test_data_loader():
    return test_data_loader
