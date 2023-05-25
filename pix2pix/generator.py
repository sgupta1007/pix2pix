import torch
import torch.nn as nn
import torch.optim as optim
from utils import create_down_sampling_layer,create_up_sampling_layer

class GeneratorPix2Pix(nn.Module):
    def __init__(self):
        super(GeneratorPix2Pix,self).__init__()
        self.layer1=create_down_sampling_layer(3,64,apply_batch_norm=False)
        self.layer2=create_down_sampling_layer(64,128)
        self.layer3=create_down_sampling_layer(128,256)
        self.layer4=create_down_sampling_layer(256,512)
        self.layer5=create_down_sampling_layer(512,512)
        self.layer6=create_down_sampling_layer(512,512)
        self.layer7=create_down_sampling_layer(512,512)
        self.layer8=create_down_sampling_layer(512,512,apply_batch_norm=False)
        self.layer7inv=create_up_sampling_layer(512,512)
        self.layer6inv=create_up_sampling_layer(512,512)
        self.layer5inv=create_up_sampling_layer(512,512)
        self.layer4inv=create_up_sampling_layer(512,512,apply_drop_out=False)
        self.layer3inv=create_up_sampling_layer(512,256,apply_drop_out=False)
        self.layer2inv=create_up_sampling_layer(256,128,apply_drop_out=False)
        self.layer1inv=create_up_sampling_layer(128,64,apply_drop_out=False)
        self.output=create_up_sampling_layer(64,3,apply_drop_out=False)

        




    
    def forward(self,x1):
        x2=self.layer1(x1)
        #print(x2.size())
        x3=self.layer2(x2)
        #print(x3.size())
        x4=self.layer3(x3)
        #print(x4.size())
        x5=self.layer4(x4)
        #print(x5.size())
        x6=self.layer5(x5)
        #print(x6.size())
        x7=self.layer6(x6)
        #print(x7.size())
        x8=self.layer7(x7)
        #print(x8.size()) 
        x8=self.layer8(x8)
        #print(x8.size())
        x7=self.layer7inv(x8)
        #print(x7.size())
        x6=self.layer6inv(x7)
        #print(x6.size())
        x5=self.layer5inv(x6)
        #print(x5.size())
        x4=self.layer4inv(x5)
        #print(x4.size())
        x3=self.layer3inv(x4)
        #print(x3.size())
        x2=self.layer2inv(x3)
        #print(x2.size())
        x1=self.layer1inv(x2)
        #print(x1.size())
        x1=self.output(x1)
        #print(x1.size())
        return x1
        

## testing script will be deleted 

m=GeneratorPix2Pix()
image=torch.randn([1,3,256,256])
output=m(image)
print(output.size())