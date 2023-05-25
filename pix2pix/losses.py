import torch
import torch.nn as nn
import torch.optim as optim

#loss for the gan object
def gan_loss(discriminator_output,generator_output,target):
    #loss function for the gan
    gan_loss_function=nn.BCELoss()
    #now we will calculate the loss
    gan_loss=gan_loss_function(torch.ones(discriminator_output.size()),discriminator_output)
    #calculate the L1 loss
    l1_loss=nn.L1Loss(generator_output,target)
    #calculate the total loss
    #return all three losses
    return gan_loss+l1_loss,gan_loss,l1_loss


#loss for the discriminator object

def disc_loss(discriminator_real_output,discriminator_gen_output):
    discriminator_loss_function=nn.BCELoss()
    real_loss=discriminator_loss_function(torch.ones(discriminator_real_output.size()),discriminator_real_output)
    gen_loss=discriminator_loss_function(torch.zeros(discriminator_gen_output.size()),discriminator_gen_output)
    return real_loss+gen_loss,real_loss,gen_loss
    
