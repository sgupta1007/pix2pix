import torch
import torch.nn as nn
import torch.optim as optim
def create_down_sampling_layer(input_size,output_size,apply_batch_norm=True,kernel_size=4,stride=2,leaky_ratio=0.2):
    layers=[]
    layers.append(nn.Conv2d(input_size,output_size,kernel_size=kernel_size,stride=stride,padding=1))
    if apply_batch_norm:
        layers.append(nn.BatchNorm2d(output_size))
    layers.append(nn.LeakyReLU(leaky_ratio))

    return nn.Sequential(*layers)


def create_up_sampling_layer(input_size,output_size,apply_drop_out=True,kernel_size=4,stride=2,drop_out_ratio=0.5):
    layers=[]
    layers.append(nn.ConvTranspose2d(input_size,output_size,kernel_size=kernel_size,stride=stride,padding=1))
    layers.append(nn.BatchNorm2d(output_size))
    if apply_drop_out:
        layers.append(nn.Dropout2d(drop_out_ratio))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)



    




    