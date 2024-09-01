import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import vgg19
import os

# Set TORCH_HOME to a new directory on the E drive
os.environ['TORCH_HOME'] = 'E:\\torch_cache'


class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels, 64, stride=2, normalize=False),
            discriminator_block(64, 128, stride=2),
            discriminator_block(128, 256, stride=2),
            discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class GrayscaleColorDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='trainval', transform=None):
        self.dataset = OxfordIIITPet(root=root, split=split, download=True, target_types='category')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        gray_img = transforms.Grayscale()(img)
        return gray_img, img


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:12].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)


if __name__ == '__main__':
    # Additional augmentations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    # Data loaders
    trainset = GrayscaleColorDataset(root='./data', split='trainval', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1)
    testset = GrayscaleColorDataset(root='./data', split='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=1)

    if torch.cuda.is_available():
        print("CUDA is available! PyTorch is using the GPU.")
    else:
        print("CUDA is not available. PyTorch is using the CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    netG = UNetGenerator(1, 3).to(device)
    netD = Discriminator(4).to(device)

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    num_epochs = 43
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            gray_images, color_images = data
            gray_images, color_images = gray_images.to(device), color_images.to(device)

            # Adversarial ground truths
            valid = torch.ones((gray_images.size(0), 1, 30, 30), requires_grad=False).to(device)
            fake = torch.zeros((gray_images.size(0), 1, 30, 30), requires_grad=False).to(device)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real images
            pred_real = netD(gray_images, color_images)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake images
            fake_images = netG(gray_images)
            pred_fake = netD(gray_images, fake_images.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            # GAN loss
            pred_fake = netD(gray_images, fake_images)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_images, color_images)

            # Perceptual loss
            loss_perceptual = criterion_perceptual(fake_images, color_images)

            # Total loss
            loss_G = loss_GAN + 100 * loss_pixel + 0.1 * loss_perceptual
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(trainloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    print('Finished Training')

    # Test the network on the test data
    dataiter = iter(testloader)
    gray_images, color_images = next(dataiter)

    # Print original and colorized images
    def show_images(gray, color, predicted):
        fig, axs = plt.subplots(3, 4, figsize=(12, 8))
        for i in range(4):
            axs[0, i].imshow(gray[i].cpu().squeeze(), cmap='gray')
            axs[0, i].set_title('Grayscale')
            axs[1, i].imshow(color[i].cpu().permute(1, 2, 0))
            axs[1, i].set_title('Original Color')
            axs[2, i].imshow(predicted[i].cpu().permute(1, 2, 0))
            axs[2, i].set_title('Predicted Color')
        plt.show()

    gray_images, color_images = gray_images.to(device), color_images.to(device)
    netG.eval()
    with torch.no_grad():
        predicted_images = netG(gray_images)
    show_images(gray_images, color_images, predicted_images)
