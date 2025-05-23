# 基于GAN的手写数字生成实验

这个实验项目实现了一个基于生成对抗网络(GAN)的手写数字生成器，使用MNIST数据集进行训练，并提供了可视化工具来展示训练过程和生成结果。

## 项目结构

```
lab1.5-gan/
├── model.py      # 定义GAN模型（生成器和判别器）
├── train.py      # GAN训练逻辑
├── visualize.py  # 可视化工具
├── main.py       # 主程序
├── README.md     # 项目说明
├── data/         # 存放MNIST数据集（自动下载）
├── images/       # 存放生成的图像
└── models/       # 存放训练好的模型
```

## 实现原理

生成对抗网络(GAN)由两部分组成：
1. **生成器(Generator)** - 学习从随机噪声生成看起来像真实手写数字的图像
2. **判别器(Discriminator)** - 学习区分真实的手写数字图像和生成器生成的假图像

这两个网络通过不断对抗训练，提高各自的能力：生成器试图生成更逼真的图像来欺骗判别器，而判别器则试图更准确地区分真假图像。

## 如何运行

1. 确保已安装必要的依赖:
   ```
   pip install torch torchvision matplotlib numpy
   ```

2. 运行主程序:
   ```
   python main.py
   ```

3. 可选的命令行参数:
   ```
   python main.py --n_epochs 100 --batch_size 64 --lr 0.0002
   ```

## 参数说明

- `--n_epochs`: 训练轮数 (默认: 200)
- `--batch_size`: 批次大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.0002)
- `--b1`: Adam优化器的beta1参数 (默认: 0.5)
- `--b2`: Adam优化器的beta2参数 (默认: 0.999)
- `--latent_dim`: 潜在空间维度 (默认: 100)
- `--sample_interval`: 采样间隔 (默认: 400)
- `--data_path`: 数据集存放路径 (默认: "./data")

## 查看结果

- 训练过程中生成的图像保存在 `images` 目录
- 训练完成后的模型权重保存在 `models` 目录

## 实验结果分析

训练结果可以从以下几个方面进行分析：
1. 生成图像的质量随着训练的进行是否有所提高？
2. 生成器和判别器的损失函数变化趋势是否符合预期？
3. 最终生成的手写数字是否逼真和多样化？
