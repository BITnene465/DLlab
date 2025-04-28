import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class GANVisualizer:
    """
    GAN可视化类，用于展示训练过程和生成结果
    """
    def __init__(self):
        # 创建保存图像的目录
        self.images_dir = "images"
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 初始化图表
        self.fig_loss, self.ax_loss = plt.subplots(figsize=(10, 5))
        self.loss_g_plot, = self.ax_loss.plot([], [], 'b', label='Generator')
        self.loss_d_plot, = self.ax_loss.plot([], [], 'r', label='Discriminator')
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('GAN Training Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True)
        
    def save_images(self, images, nrow=10, ncol=10, batches_done=None):
        """
        保存生成的图像
        """
        # 将图像转换为numpy数组
        images_cpu = images.detach().cpu()
        
        # 创建网格图像
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        
        # 绘制图像
        counter = 0
        for i in range(nrow):
            for j in range(ncol):
                if counter < len(images_cpu):
                    # 数据范围从[-1, 1]转换为[0, 1]
                    img = images_cpu[counter].squeeze().numpy() * 0.5 + 0.5
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].axis('off')
                    counter += 1
        
        # 保存图像
        filename = f"gan_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" if batches_done is None else f"gan_generated_{batches_done}.png"
        file_path = os.path.join(self.images_dir, filename)
        plt.savefig(file_path)
        plt.close(fig)
        print(f"Saved generated images to {file_path}")
        
    def update_loss_plot(self, g_losses, d_losses):
        """
        更新损失图表
        """
        iterations = np.arange(len(g_losses))
        self.loss_g_plot.set_data(iterations, g_losses)
        self.loss_d_plot.set_data(iterations, d_losses)
        
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # 保存损失图表
        filename = f"gan_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(self.images_dir, filename)
        self.fig_loss.savefig(file_path)
        print(f"Saved loss plot to {file_path}")
    
    def display_grid_images(self, images, nrow=10, ncol=10):
        """
        显示生成的图像网格
        """
        # 将图像转换为numpy数组
        images_cpu = images.detach().cpu()
        
        # 创建网格图像
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        
        # 绘制图像
        counter = 0
        for i in range(nrow):
            for j in range(ncol):
                if counter < len(images_cpu):
                    # 数据范围从[-1, 1]转换为[0, 1]
                    img = images_cpu[counter].squeeze().numpy() * 0.5 + 0.5
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].axis('off')
                    counter += 1
        
        plt.tight_layout()
        plt.show()
