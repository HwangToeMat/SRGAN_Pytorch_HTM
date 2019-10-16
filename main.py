import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SRGAN import G_Net, D_Net
from LossFunction import GeneratorLoss, DiscriminatorLoss
from dataset_h5 import Read_dataset_h5
import pytorch_ssim

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRGAN")
parser.add_argument("--batchSize", type=int, default=64)
parser.add_argument("--nEpochs", type=int, default=100)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument("--gpus", default="0", type=str)

def main():
    global opt, G_Net, D_Net , G_optim, D_optim
    epoch = 1
    opt = parser.parse_args() # opt < parser
    print(opt)

    print("===> Setting GPU")
    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus # set gpu
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed) # set seed
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True # find optimal algorithms for hardware

    print("===> Loading datasets")
    train_set = Read_dataset_h5("data/train.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
        batch_size=opt.batchSize, shuffle=True) # read to DataLoader

    print("===> Building model")
    G_Net = G_Net()
    D_Net = D_Net()
    G_Loss = GeneratorLoss()
    D_Loss = DiscriminatorLoss()

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            G_Net.load_state_dict(checkpoint['G_Net_state_dict'])
            D_Net.load_state_dict(checkpoint['D_Net_state_dict'])
            epoch = checkpoint['epoch'] + 1 # load model
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    if cuda:
        G_Net = G_Net.cuda()
        D_Net = D_Net.cuda()
        G_Loss = G_Loss.cuda()
        D_Loss = D_Loss.cuda() # set model&loss for use gpu

    print("===> Setting Optimizer")
    G_optim = optim.Adam(G_Net.parameters())
    D_optim = optim.Adam(D_Net.parameters())

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
            D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
            print("===> Setting Pretrained Optimizer")

    print("=> start epoch '{}'".format(epoch))
    print("===> Training")
    for epoch_ in range(epoch, opt.nEpochs + 1):
        print("===>  Start epoch {} #################################################################".format(epoch_))
        G_Net.train()
        D_Net.train()
        for _, (input, label) in enumerate(training_data_loader):
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
                print("===> Epoch[[{}]({}/{})]: D_Loss : {:.10f}, G_Loss : {:.10f}, SSIM : {:.10f}".format(epoch_, _, len(training_data_loader), d_loss, g_loss, pytorch_ssim.ssim(HR, fake_img)))

        model_out_path = "checkpoint/" + "SRGAN_Adam_epoch_{}.tar".format(epoch_)
        if not os.path.exists("checkpoint/"):
            os.makedirs("checkpoint/")
        torch.save({
                'epoch': epoch_,
                'G_Net_state_dict': G_Net.state_dict(),
                'D_Net_state_dict': D_Net.state_dict(),
                'G_optim_state_dict': G_optim.state_dict(),
                'D_optim_state_dict': D_optim.state_dict()
                }, model_out_path)
        print("Checkpoint has been saved to the {}".format(model_out_path))

if __name__ == "__main__":
    main()
