import torch
import torch.nn as nn
import torch.optim as optim
from utils import create_down_sampling_layer

#70*70 discriminator as per the paper architecture with deviation that i am supplying 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1=create_down_sampling_layer(6,64,apply_batch_norm=False)
        self.layer2=create_down_sampling_layer(64,128)
        self.layer3=create_down_sampling_layer(128,256)
        #self.padd_1=nn.ZeroPad2d(1)
        self.layer4=create_down_sampling_layer(256,512,apply_batch_norm=True,kernel_size=4,stride=2)
        #self.layer5=last_layer_70_by_70_discriminator
        self.layer5=nn.Conv2d(512,1,kernel_size=1)
        #add a Tanh  layer
        self.layer6=nn.Tanh()

        
    #in forward logic we will concatenate the inputs with dim=1   
    def forward(self,x,y):
        x_y=self.layer1(torch.cat((x,y),dim=1))
        #print(x_y.size())
        x_y=self.layer2(x_y)
        #print(x_y.size())
        x_y=self.layer3(x_y)
        #print(x_y.size())
        #x_y=self.padd_1(x_y)
        #print(x_y.size())
        x_y=self.layer4(x_y)
        #print(x_y.size())
        x_y=self.layer5(x_y)
        #print(x_y.size())
        x_y=self.layer6(x_y)
        return x_y

## testing script will be deleted 

m=Discriminator()
image=torch.randn([1,3,256,256])
cimage=torch.rand([1,3,256,256])
output=m(image,cimage)
#print(output.size())