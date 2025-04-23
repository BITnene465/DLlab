import sys
import os
from pathlib import Path
# 设置项目根目录= Path(__file__).parent.parent
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from myTransformer import Transformer
from myTokenizer import Tokenizer
from Datasets import E2EDataset
from train_utils import train_one_epoch, validate, set_device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time 
import json


def train_transformer(model, train_dataloader, valid_grouped_data, optimizer, criterion, device, n_epochs, save_dir, patience=3, clip=1.0, best_model_path=None):
    """ 
    Args:
        model: 模型实例
        train_dataloader: 训练数据加载器
        valid_grouped_data: 验证集的分组数据
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        n_epochs: 训练轮数
        save_dir: 模型保存目录
        patience: 早停耐心值
        best_model_path: 最佳模型路径，如果提供则加载模型继续训练
    
    Returns:
        model: 训练后的模型
        train_losses: 训练损失历史
        valid_losses: 验证损失历史
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 如果提供了模型路径，则加载模型
    if best_model_path is not None and os.path.exists(best_model_path):
        model.load_model(best_model_path)
        print(f"加载预训练模型: {best_model_path}")
    
    train_losses = []
    valid_bleus = []  # 添加BLEU记录
    best_valid_bleu = 0
    patience_counter = 0
    
    for epoch in range(n_epochs):
        start_time = time.time()
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, clip)
        train_losses.append(train_loss)
        # 在验证集上评估
        valid_bleu = validate(model, valid_grouped_data, max_tgt_len, device)
        valid_bleus.append(valid_bleu)  # 记录BLEU分数
        # 计算耗时
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        # 检查是否是最佳模型
        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            patience_counter = 0
            model_path = os.path.join(save_dir, f'model_epoch{epoch+1}_bleu{valid_bleu:.4f}.pt')
            model.save_model(model_path)
            best_model_path = model_path
            print(f"发现更好的模型，已保存到 {model_path}")
        else:
            patience_counter += 1
        print(f'Epoch: {epoch+1:02} | 耗时: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\t训练损失: {train_loss:.4f}')
        print(f'\t验证集 BLEU-4: {valid_bleu:.4f}')
        
        # 早停
        if patience_counter >= patience:
            print(f"连续 {patience} 个epoch验证损失没有改善，提前停止训练")
            break
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'valid_bleus': valid_bleus,  # 添加BLEU历史
        'best_valid_bleu': best_valid_bleu,
        'best_model_path': best_model_path,
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # 保存最后的模型
    model_path = os.path.join(save_dir, 'last_model.pt')
    model.save_model(model_path)
    
    # 加载最佳模型
    if best_model_path is not None:
        model.load_model(best_model_path)
    
    return model, train_losses, valid_bleus


if __name__ == "__main__":
    # 超参数设置
    max_src_len = 50
    max_tgt_len = 50
    batch_size = 128
    epochs = 10
    lr = 0.001
    weight_decay = 1e-4
    patience = 10  # 早停耐心值
    clip = 5.0

    # tokenizer & model
    tokenizer_path = ROOT_DIR / "tokenizers/e2e_tokenizer.json"
    tokenizer = Tokenizer.load(tokenizer_path)
    
    # 数据集
    dataset_dir = ROOT_DIR / "e2e_dataset"
    train_dataset = E2EDataset(
        csv_file=dataset_dir / "trainset.csv",
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        is_test=False,
    )
    val_dataset = E2EDataset(
        csv_file=dataset_dir / "devset.csv",
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        is_test=False,
    )
    
    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_grouped_data = val_dataset.get_grouped_data()  # 用于多参考评估的分组数据

    # train
    device = set_device()
    model = Transformer(
        tokenizer=tokenizer,
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    ).to(device)  
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    save_dir = os.path.join(ROOT_DIR, "saved_models")
    best_model_path = None   # 用于继续训练
    
    _, _, _ = train_transformer(
        model=model,
        train_dataloader=train_dataloader,
        valid_grouped_data=val_grouped_data,  # 用于验证的分组数据
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=epochs,
        save_dir=save_dir,
        clip=clip,
        patience=patience,
        best_model_path=best_model_path,  # 用于继续训练
    )
    
    
