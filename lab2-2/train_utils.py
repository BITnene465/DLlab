import torch
import numpy as np
import random
import os
from typing import Optional, Dict, Any
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch
from torch.utils.data import DataLoader 


def set_seed(seed: int = 42):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

def create_padding_mask(seq: list[int], pad_idx=0):
    """创建padding掩码（用于屏蔽<PAD>标记）"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2).float()
    return mask

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
    
def validate(model, grouped_data: dict, max_tgt_len, device):
    """使用多参考评估模型, batch_size=1"""
    model.eval()
    start_symbol_id = model.tokenizer.BOS_ID
    end_symbol_id = model.tokenizer.EOS_ID
    # 用于计算BLEU的参考和假设
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for _, data in tqdm(grouped_data.items(), desc="validating..."):
            src_ids = data['src_ids'].unsqueeze(0).to(device)  # 添加batch维度
            src_mask = None
            # 生成序列
            predicted_ids = model.generate(src=src_ids, src_mask=src_mask, max_len=max_tgt_len, start_symbol_id=start_symbol_id, end_symbol_id=end_symbol_id, temperature=1.0)
            predicted_ids = predicted_ids.squeeze(0)  # 去掉batch维度
            
            # 将参考和预测添加到评估列表
            tgt_ids_list = data['tgt_ids_list']
            for i in range(len(tgt_ids_list)):
                if end_symbol_id in tgt_ids_list[i]:    # 到 EOS 截断
                    eos_index = tgt_ids_list[i].index(end_symbol_id)
                    tgt_ids_list[i] = tgt_ids_list[i][:eos_index]
            references.append(tgt_ids_list)
            hypotheses.append(predicted_ids)
           
    # 计算BLEU-4
    bleu4 = calculate_bleu4(references, hypotheses)
    return bleu4

def train_one_epoch(model, dataloader: DataLoader, optimizer, criterion, device, teacher_forcing: bool=True, clip=1.0):
    """
    Args:
        model: 模型实例
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        clip: 梯度裁剪阈值
        teacher_forcing: 是否启用teacher forcing
    
    Returns:
        epoch_loss: 平均损失
    """
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc=f"training... "):
        # 准备数据
        src_ids = batch['src_ids'].to(device)
        tgt_ids = batch['tgt_ids'].to(device)
        src_padding_mask = create_padding_mask(src_ids).to(device)
        tgt_padding_mask = create_padding_mask(tgt_ids).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
        
        # 训练
        optimizer.zero_grad()     
        
        # todo: 前向传播需要修改
        output = model(src=src_ids, tgt=tgt_ids, src_mask=src_padding_mask, tgt_mask=tgt_mask)
        
        output = output.view(-1, output.shape[-1])
        tgt_ids = tgt_ids.view(-1)
        loss = criterion(output, tgt_ids)    # 计算损失 (已经在 criterion 中忽略 PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  #! 梯度裁剪
        optimizer.step()
        # 累加 loss
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


