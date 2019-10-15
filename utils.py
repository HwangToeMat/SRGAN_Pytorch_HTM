import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_ssim


def Train(dataloader, G_Net, D_Net , G_optim, D_optim, G_Loss, D_Loss, epoch_):
    G_Net.train()
    D_Net.train()
    for _, (input, label) in enumerate(dataloader):
        HR = Variable(label)/255
        LR = Variable(input)/255
        if torch.cuda.is_available():
            HR = HR.cuda()
            LR = LR.cuda()
        fake_img = G_Net(LR)

        # Train Discriminator model
        D_Net.zero_grad()
        real_rate = D_Net(HR)
        fake_rate = D_Net(fake_img)
        d_loss = D_Loss(fake_rate, real_rate)
        d_loss.backward(retain_graph=True)
        D_optim.step()

        # Train Generator model
        G_Net.zero_grad()
        g_loss = G_Loss(fake_rate, fake_img, HR)
        g_loss.backward()
        G_optim.step()

        # loss 출력
        if _%10 == 0:
            print("===> Epoch[[{}]({}/{})]: D_Loss : {:.10f}, G_Loss : {:.10f}, SSIM : {:.10f}".format(epoch_, _, len(dataloader), d_loss, g_loss, pytorch_ssim.ssim(HR, fake_img)))

def save_checkpoint(epoch, G_Net, D_Net , G_optim, D_optim):
    model_out_path = "checkpoint/" + "SRGAN_Adam_epoch_{}.tar".format(epoch)
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save({
            'epoch': epoch,
            'G_Net_state_dict': G_Net.state_dict(),
            'D_Net_state_dict': D_Net.state_dict(),
            'G_optim_state_dict': G_optim.state_dict(),
            'D_optim_state_dict': D_optim.state_dict()
            }, model_out_path)
    print("Checkpoint has been saved to the {}".format(model_out_path))
