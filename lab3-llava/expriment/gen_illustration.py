import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
import random

def create_dataset_overview(
    image_dir, 
    output_path="dataset_overview.png", 
    sample_ratio=1.0, 
    max_images=12, 
    figsize=(12, 10), 
    dpi=300,
    overlap_factor=0.15,
    jitter_factor=0.05
):
    # 获取所有图片路径
    image_dir = Path(image_dir)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
    
    if not image_files:
        raise ValueError(f"在 {image_dir} 中没有找到图片文件")
    
    # 根据采样比例和上限抽取图片
    num_images = len(image_files)
    num_to_sample = min(int(num_images * sample_ratio), max_images)
    
    if num_to_sample < num_images:
        sampled_images = random.sample(image_files, num_to_sample)
    else:
        sampled_images = image_files[:num_to_sample]
    
    # 计算网格布局
    n = len(sampled_images)
    cols = int(np.ceil(np.sqrt(n * 1.2)))
    rows = int(np.ceil(n / cols))
    
    # 创建画布
    fig = plt.figure(figsize=figsize, facecolor='white')
    # 加载并显示每张图片，带有错位和重叠效果
    for i, img_path in enumerate(sampled_images):
        # 计算基础位置
        row, col = i // cols, i % cols
        base_x = col / cols
        base_y = 1.0 - (row + 1) / rows
        # 添加随机偏移
        jitter_x = (random.random() - 0.5) * jitter_factor
        jitter_y = (random.random() - 0.5) * jitter_factor
        # 计算最终位置和大小
        width = 1.0 / cols * (1 + overlap_factor)
        height = 1.0 / rows * (1 + overlap_factor)
        rect = [base_x + jitter_x, base_y + jitter_y, width, height]
        # 创建子图
        ax = fig.add_axes(rect)
        # 显示图片
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_title(f"{img_path.stem}", fontsize=8)
        ax.axis('off')
        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5) 
    plt.suptitle(f"dataset overview", fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"已创建数据集概览图并保存至: {output_path}")
    return output_path

if __name__ == "__main__":
    image_directory = "ourdata/images"
    create_dataset_overview(
        image_directory,
        output_path="dataset_overview.png",
        sample_ratio=1,
        max_images=9,
        overlap_factor=0.2,
        jitter_factor=0.08
    )