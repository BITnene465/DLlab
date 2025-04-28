import torch
import argparse
from train import GANTrainer
from visualize import GANVisualizer
import os

def main():
    parser = argparse.ArgumentParser(description="MNIST GAN")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--data_path", type=str, default="./data", help="path to store the dataset")
    opt = parser.parse_args()
    
    # 打印配置
    print("配置参数:")
    print(f"- 训练轮数: {opt.n_epochs}")
    print(f"- 批次大小: {opt.batch_size}")
    print(f"- 学习率: {opt.lr}")
    print(f"- Adam β1: {opt.b1}")
    print(f"- Adam β2: {opt.b2}")
    print(f"- 潜在空间维度: {opt.latent_dim}")
    print(f"- 采样间隔: {opt.sample_interval}")
    print(f"- 数据路径: {opt.data_path}")
    
    # 创建训练器和可视化器
    trainer = GANTrainer(
        data_path=opt.data_path,
        batch_size=opt.batch_size,
        latent_dim=opt.latent_dim,
        lr=opt.lr,
        b1=opt.b1,
        b2=opt.b2,
        n_epochs=opt.n_epochs,
        sample_interval=opt.sample_interval
    )
    
    visualizer = GANVisualizer()
    
    # 开始训练
    print("开始训练...")
    for gen_imgs, batches_done in trainer.train():
        # 每隔一段时间保存生成的图像
        visualizer.save_images(gen_imgs, nrow=5, ncol=5, batches_done=batches_done)
        
        # 更新损失图表
        visualizer.update_loss_plot(trainer.g_losses, trainer.d_losses)
    
    # 训练结束后生成最终结果
    print("\n训练完成！生成最终图像...")
    final_images = trainer.sample_images(n_row=10, n_col=10)
    visualizer.save_images(final_images, nrow=10, ncol=10)
    
    # 显示最终的损失曲线
    visualizer.update_loss_plot(trainer.g_losses, trainer.d_losses)
    
    print("\nGAN训练和图像生成完成!")
    print("- 生成的图像保存在 'images' 目录")
    print("- 训练的模型保存在 'models' 目录")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
