from discriminator import Discriminator
from generator import GeneratorPix2Pix
from data_utils import get_train_data_loader,get_test_data_loader


import torch
import torch.nn as nn
import torch.optim as optim

disc=Discriminator()
gen=GeneratorPix2Pix()

#constants required for training
N_epochs=10
lr=0.002
batch_size=1

#loss and optimizer
loss=nn.BCELoss()
l1loss=nn.L1Loss()
optimD=optim.Adam(disc.parameters())
optimG=optim.Adam(gen.parameters())

#loading  the datasets
train_data_loader=get_train_data_loader()
test_data_loader=get_test_data_loader()




#labels for datasets
real_label=1.0 #for real images
fake_label=0.0 # for fake images

total_losses=[]
disc_losses=[]
gen_losses=[]

for iters in range(N_epochs):
    for (i,data) in enumerate(train_data_loader):
        #train discriminator
        real_image_x=data[0]
        real_image_y=data[1]
        disc_real_output=disc(real_image_x,real_image_y)
        disc_real_loss=loss(disc_real_output,torch.ones(disc_real_output.size()))

        #noise will be added to generator
        noise=torch.randn(real_image_x.size())

        #backprop
        optimD.zero_grad()
        disc_real_loss.backward()
        optimD.step()
        #now generate the fake images

        fake_images=gen(noise)

        #calculate the discriminator output with respect to fake_images

        disc_fake_output=disc(fake_images,real_image_x)
        disc_fake_loss=loss(disc_fake_output,torch.zeros(disc_fake_output.size()))

        #backprop
        optimD.zero_grad()
        disc_fake_loss.backward()
        optimD.step()


        #calculate total disc loss
        total_disc_loss=disc_fake_loss+disc_real_loss


        #train generator

        fake_images=gen(noise)
        disc_gen_out=disc(fake_images,real_image_x)

        #calculate the generator loss
        gen_gan_loss=loss(disc_gen_out,torch.ones(disc_real_output.size()))
        gen_l1_loss=l1loss(fake_images,real_image_y)



        total_gen=gen_gan_loss+gen_l1_loss

        #backprop generator
        optimG.zero_grad()
        total_gen.backward()
        optimG.step()

        #total losses calculation
        total_loss=total_disc_loss+total_gen
        total_losses.append(total_loss)
        gen_losses.append(total_gen)
        disc_losses.append(total_disc_loss)

    print(total_losses)
        
        



















