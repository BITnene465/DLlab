import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Generator, Discriminator
import os
import numpy as np

class GANTrainer:
    """
    GAN训练类，管理训练过程
    """
    def __init__(self, data_path="./data", batch_size=64, latent_dim=100, lr=0.0002, b1=0.5, b2=0.999, n_epochs=200, sample_interval=400, device=None):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.sample_interval = sample_interval
        
        # 确定使用的设备
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 创建数据目录
        os.makedirs(data_path, exist_ok=True)
        
        # 加载MNIST数据集
        self.dataloader = self._get_data_loader(data_path)
        
        # 初始化模型
        self.generator = Generator(latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 初始化损失函数和优化器
        self.adversarial_loss = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        # 用于记录损失
        self.g_losses = []
        self.d_losses = []
        
    def _get_data_loader(self, data_path):
        """
        获取MNIST数据加载器
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def train(self):
        """
        训练GAN模型
        """
        # 用于记录样本
        batches_done = 0
        
        # 开始训练循环
        for epoch in range(self.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                # 配置真实图像和标签
                real_imgs = imgs.to(self.device)
                batch_size = imgs.shape[0]
                
                # 创建标签
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)
                
                # -----------------
                # 训练生成器
                # -----------------
                self.optimizer_G.zero_grad()
                
                # 生成随机噪声
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                
                # 生成一批假图像
                gen_imgs = self.generator(z)
                
                # 计算生成器损失
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                
                # 反向传播和优化
                g_loss.backward()
                self.optimizer_G.step()
                
                # -----------------
                # 训练判别器
                # -----------------
                self.optimizer_D.zero_grad()
                
                # 计算真实图像的损失
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                
                # 计算假图像的损失
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                
                # 综合损失
                d_loss = (real_loss + fake_loss) / 2
                
                # 反向传播和优化
                d_loss.backward()
                self.optimizer_D.step()
                
                # 记录损失
                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item())
                
                # 输出训练状态
                if i % 100 == 0:
                    print(
                        f"[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(self.dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                    )
                
                batches_done = epoch * len(self.dataloader) + i
                
                # 每训练一定步骤，保存样本
                if batches_done % self.sample_interval == 0:
                    # 返回当前批次生成的图像和训练步骤
                    yield gen_imgs, batches_done
        
        print("Training complete!")
        # 保存模型
        self._save_models()
        
    def _save_models(self):
        """
        保存训练好的模型
        """
        os.makedirs("models", exist_ok=True)
        torch.save(self.generator.state_dict(), "models/generator.pth")
        torch.save(self.discriminator.state_dict(), "models/discriminator.pth")
        print("Models saved!")
    
    def sample_images(self, n_row=10, n_col=10):
        """
        生成样本图像
        """
        z = torch.randn(n_row * n_col, self.latent_dim, device=self.device)
        gen_imgs = self.generator(z)
        return gen_imgs
