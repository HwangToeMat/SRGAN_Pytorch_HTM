# SRGAN_Pytorch_HTM
## Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

## 모델 구조

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/assets/img/thumbnail/pr-3-1.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

그림과 같이 새로운 데이터(SR)을 생성하는 Generator model과 그 데이터를 판단하는 Discriminator model로 이루어졌다. Super-Resolution에서 GAN의 구조를 처음으로 도입하였다.

## Usage

```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--cuda]
               [--threads THREADS] [--pretrained PRETRAINED] [--gpus GPUS]

PyTorch SRGAN

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
  --nEpochs NEPOCHS
  --cuda
  --threads THREADS
  --pretrained PRETRAINED
  --gpus GPUS
```

## Data augmentation

### flip

```python
for flip in [0,1]:
    if flip == 0:
        image_f = image
    else:
        image_f = cv2.flip(image,1)
```

### rotate

```python
def img_rotate(img, degree):
    height, width = img.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), 90*degree, 1)
    if degree == 1 or degree == 3:
        dst = cv2.warpAffine(img, matrix, (height, width))
    else:
        dst = cv2.warpAffine(img, matrix, (width, height))
    return dst
```

### downsize

```python
def img_downsize(img, ds):
    dst = cv2.resize(img, dsize=(0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_LINEAR)
    return dst
```

### crop image

```python
def sub_img(input, label, i_size = 33, l_size = 21, stride = 14):
    sub_ipt = []
    sub_lab = []
    pad = abs(i_size-l_size)//2
    for h in range(0, input.shape[0] - i_size + 1, stride):
        for w in range(0, input.shape[1] - i_size + 1, stride):
            sub_i = input[h:h+i_size,w:w+i_size]
            sub_l = label[h + pad :h + pad + l_size,w + pad :w + pad + l_size]
            sub_i = sub_i.reshape(1, i_size,i_size)
            sub_l = sub_l.reshape(1, l_size,l_size)
            sub_ipt.append(sub_i)
            sub_lab.append(sub_l)
    return sub_ipt, sub_lab
```

### down scale

```python
def zoom_img(img, scale):
    label = img.astype('float') / 255
    temp_input = cv2.resize(label, dsize=(0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
    input = cv2.resize(temp_input, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return input, label
```


## Model

* Generator Model

```python
class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.ResidualBlock1 = ResidualBlock(64)
        self.ResidualBlock2 = ResidualBlock(64)
        self.ResidualBlock3 = ResidualBlock(64)
        self.ResidualBlock4 = ResidualBlock(64)
        self.ResidualBlock5 = ResidualBlock(64)
        self.output_residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(64, 64 * 2 ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.pixel_shuffle2 = nn.Sequential(
            nn.Conv2d(64, 64 * 2 ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.output =  nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        input = self.input(x)
        ResidualBlock1 = self.ResidualBlock1(input)
        ResidualBlock2 = self.ResidualBlock2(ResidualBlock1)
        ResidualBlock3 = self.ResidualBlock3(ResidualBlock2)
        ResidualBlock4 = self.ResidualBlock4(ResidualBlock3)
        ResidualBlock5 = self.ResidualBlock5(ResidualBlock4)
        output_residual = self.output_residual(ResidualBlock5)
        pixel_shuffle = self.pixel_shuffle(output_residual + input)
        pixel_shuffle2 = self.pixel_shuffle2(pixel_shuffle)
        output = self.output(pixel_shuffle2)
        output = (output+1)/2
        return output
```

* Discriminator Model

```python
class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.Net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Net(x)
        return x.squeeze()
```

## Train

```python
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
            d_loss.backward()
            D_optim.step()

            # Train Generator model
            G_Net.zero_grad()
            g_loss = G_Loss(fake_rate, fake_img, HR)
            g_loss.backward()
            G_optim.step()
```

## loss function

* GeneratorLoss(MSE_loss  + VGG_loss + Adversarial_loss)

```python
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        vgg_loss = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in vgg_loss.parameters():
            param.requires_grad = False
        self.vgg_loss = vgg_loss
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.BCELoss()

    def forward(self, fake_rate, SR, HR):
        # MSE Loss
        MSE_loss = self.mse_loss(SR, HR)
        # VGG Loss
        VGG_loss = self.mse_loss(self.vgg_loss(SR), self.vgg_loss(HR))
        # Adversarial Loss
        Adversarial_loss = self.cross_entropy(fake_rate,torch.ones(fake_rate.size(0)).cuda())
        return MSE_loss + 6e-3 * VGG_loss + 1e-3 * Adversarial_loss
```

* DiscriminatorLoss(Fake_img Correct Rate + Real_img Correct Rate)

```python
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.cross_entropy = nn.BCELoss()

    def forward(self, fake_rate, real_rate):
        # Fake_img Correct Rate
        Fake_img_CR = self.cross_entropy(fake_rate,torch.zeros(fake_rate.size(0)).cuda())
        # Real_img Correct Rate
        Real_img_CR = self.cross_entropy(real_rate,torch.ones(real_rate.size(0)).cuda())
        return Fake_img_CR + Real_img_CR
```

## Test

### Parameters

```
batchSize =64
cuda = True
lr = 0.001
nEpochs = 20
optimizer= 'Adam'
```
-> 30 hours for train

### Result[LR(left), SR(right)]​

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/SRGAN/image8.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

원본사진에서 매우 작은 부분을 확대한 이미지인 만큼 엄청나게 선명한 이미지는 아니지원 원본(왼쪽)에 비해 결과물(오른쪽)이 경계선부분이 훨씬 매끄러운것을 알 수 있다.

