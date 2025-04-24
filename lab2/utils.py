# 预处理阶段，转换 mr 字段的格式
def reformat_mr(text):
    """
    将 "name[Blue Spice], eatType[coffee shop], area[city centre]" 格式
    转换为 "[name = Blue Spice] , [eatType = coffee shop] , [area = city centre]" 格式
    
    Args:
        text: 输入文本字符串
        
    Returns:
        重新格式化的字符串
    """
    # 按逗号分割不同的属性
    parts = text.split(', ')
    result = []
    for part in parts:
        # 找到属性名称和值
        attr_end = part.find('[')
        value_start = attr_end + 1
        value_end = part.rfind(']')
    
        if attr_end != -1 and value_end != -1:
            attr_name = part[:attr_end]
            attr_value = part[value_start:value_end]
            # 构建新格式
            new_format = f"[{attr_name} = {attr_value}]"
            result.append(new_format)
    
    return " , ".join(result)



# 计算 bleu4分数
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
def calculate_bleu4(references, hypotheses):
    """
    计算BLEU-4分数
    
    Args:
        references: 参考翻译（真实目标）列表的列表，每个参考是token列表
        hypotheses: 模型生成的翻译（预测）列表，每个预测是token列表
    
    Returns:
        bleu4: BLEU-4分数
    """
    # 使用平滑函数避免0分
    smoothie = SmoothingFunction().method1
    # 计算BLEU-4
    return corpus_bleu(
        references, 
        hypotheses, 
        weights=(0.25, 0.25, 0.25, 0.25), 
        smoothing_function=smoothie
    )

# 绘图相关逻辑
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_train_curve(train_losses, valid_bleus, save_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(valid_bleus, label='Validation BLEU-4', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-4')
    plt.title('Validation BLEU-4 Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))


def plot_attention(attention, source_tokens, target_tokens, title=None, filename=None):
    """
    绘制注意力热力图
    Args:
        attention: 注意力权重矩阵 (tgt_len, src_len)
        source_tokens: 源序列token
        target_tokens: 目标序列token
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    attention = attention.cpu().detach().numpy()
    
    # 修剪特殊token
    if "<pad>" in source_tokens:
        first_pad = source_tokens.index("<pad>")
        source_tokens = source_tokens[:first_pad]
        attention = attention[:, :first_pad]
    
    if "<pad>" in target_tokens:
        first_pad = target_tokens.index("<pad>")
        target_tokens = target_tokens[:first_pad]
        attention = attention[:first_pad, :]
    
    plt.imshow(attention, cmap='viridis')
    plt.colorbar()
    
    # 设置x和y轴标签
    plt.xticks(np.arange(len(source_tokens)), source_tokens, rotation=90)
    plt.yticks(np.arange(len(target_tokens)), target_tokens)
    
    plt.tight_layout()
    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



