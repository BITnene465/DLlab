import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from seq2seq import Seq2SeqModel
from build_vocab import get_tokenizer_from_file
from datasets import E2EDataset


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

def train_one_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    """
    Args:
        model: 模型实例
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        clip: 梯度裁剪阈值
    
    Returns:
        epoch_loss: 平均损失
    """
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="training"):
        # 准备数据
        src_ids = batch['src_ids'].to(device)
        tgt_ids = batch['tgt_ids'].to(device)
        src_len = batch['src_len'].cpu()  # pack_padded_sequence 要求有效长度必须在 CPU 上
        # 训练
        optimizer.zero_grad()     
        output, _, _ = model(
            input_ids=src_ids,
            valid_src_len=src_len,
            max_tgt_len=tgt_ids.size(1),
            target_ids=tgt_ids
        )
        output = output.view(-1, output.shape[-1])
        tgt_ids = tgt_ids.view(-1)
        loss = criterion(output, tgt_ids)    # 计算损失 (已经在 criterion 中忽略 PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  #! 梯度裁剪
        optimizer.step()
        # 累加 loss
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    val_loss = 0
    
    # 用于计算BLEU的参考和假设
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="validating"):
            # 准备数据
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_len = batch['src_len'].cpu()
            
            # 使用自回归方式生成（不使用teacher forcing）
            output, _, _ = model(
                input_ids=src_ids,
                valid_src_len=src_len,
                max_tgt_len=tgt_ids.size(1),
                target_ids=None  # 不使用teacher forcing
            )
            
            # 计算损失
            output_flatten = output.view(-1, output.shape[-1])
            tgt_ids_flatten = tgt_ids.view(-1)
            loss = criterion(output_flatten, tgt_ids_flatten)
            val_loss += loss.item()
            
            # 获取预测结果
            _, predictions = torch.max(output, dim=2)
            
            # 处理每个样本的预测和参考
            batch_size = predictions.size(0)
            for i in range(batch_size):
                # 处理预测序列，移除特殊token
                pred_tokens = []
                for token_id in predictions[i].cpu().numpy():
                    # 如果是EOS或PAD，停止添加token
                    if token_id == model.tokenizer.eos_token_id or token_id == model.tokenizer.pad_token_id:
                        break
                    # 跳过SOS和PAD token
                    if token_id != model.tokenizer.sos_token_id and token_id != model.tokenizer.pad_token_id:
                        token = model.tokenizer.vocab.idx2token.get(token_id, "<UNK>")
                        pred_tokens.append(token)
                
                # 处理参考序列，移除特殊token
                ref_tokens = []
                for token_id in tgt_ids[i].cpu().numpy():
                    # 如果是EOS或PAD，停止添加token
                    if token_id == model.tokenizer.eos_token_id or token_id == model.tokenizer.pad_token_id:
                        break
                    # 跳过SOS和PAD token
                    if token_id != model.tokenizer.sos_token_id and token_id != model.tokenizer.pad_token_id:
                        token = model.tokenizer.vocab.idx2token.get(token_id, "<UNK>")
                        ref_tokens.append(token)
                
                # 添加到列表中用于计算BLEU
                hypotheses.append(pred_tokens)
                references.append([ref_tokens]) 
    
    # 计算BLEU-4
    bleu4 = calculate_bleu4(references, hypotheses)
    
    return val_loss / len(dataloader), bleu4

def train(model, train_dataloader, valid_dataloader, optimizer, criterion, device, 
          n_epochs, save_dir, patience=3, clip=1.0, best_model_path=None):
    """ 
    Args:
        model: 模型实例
        train_dataloader: 训练数据加载器
        valid_dataloader: 验证数据加载器
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
    valid_losses = []
    valid_bleus = []  # 添加BLEU记录
    best_valid_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        start_time = time.time()
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, clip)
        train_losses.append(train_loss)
        # 在验证集上评估
        valid_loss, valid_bleu = validate(model, valid_dataloader, criterion, device)
        valid_losses.append(valid_loss)
        valid_bleus.append(valid_bleu)  # 记录BLEU分数
        # 计算耗时
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        # 检查是否是最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            model_path = os.path.join(save_dir, f'model_epoch{epoch+1}_loss{valid_loss:.4f}_bleu{valid_bleu:.4f}.pt')
            model.save_model(model_path)
            best_model_path = model_path
            print(f"发现更好的模型，已保存到 {model_path}")
        else:
            patience_counter += 1
        print(f'Epoch: {epoch+1:02} | 耗时: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\t训练损失: {train_loss:.4f}')
        print(f'\t验证损失: {valid_loss:.4f} | BLEU-4: {valid_bleu:.4f}')
        
        # 早停
        if patience_counter >= patience:
            print(f"连续 {patience} 个epoch验证损失没有改善，提前停止训练")
            break
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_bleus': valid_bleus,  # 添加BLEU历史
        'best_model_path': best_model_path,
        'best_valid_loss': best_valid_loss
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # 加载最佳模型
    if best_model_path is not None:
        model.load_model(best_model_path)
    
    return model, train_losses, valid_losses


if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    # 数据集路径
    dataset_dir = "./e2e_dataset"
    train_path = os.path.join(dataset_dir, "trainset.csv")
    valid_path = os.path.join(dataset_dir, "devset.csv")
    
    # 超参数
    vocab_path = "./vocab.json"
    embed_size = 512
    batch_size = 64
    learning_rate = 0.00005
    weight_decay = 1e-5
    clip = 5
    n_epochs = 30
    patience = 10
    max_src_len = 40
    max_tgt_len = 40
    save_dir = "./saved_models"
    
    # 加载词汇表
    tokenizer = get_tokenizer_from_file(vocab_path=vocab_path)
    
    # 创建数据集和数据加载器
    train_dataset = E2EDataset(train_path, tokenizer, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    valid_dataset = E2EDataset(valid_path, tokenizer, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: {
            'src_ids': torch.stack([item['src_ids'] for item in x]),
            'tgt_ids': torch.stack([item['tgt_ids'] for item in x]),
            'src_len': torch.tensor([item['src_len'] for item in x]),
            'tgt_len': torch.tensor([item['tgt_len'] for item in x]),
            'src_text': [item['src_text'] for item in x],
            'tgt_text': [item['tgt_text'] for item in x]
        }
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: {
            'src_ids': torch.stack([item['src_ids'] for item in x]),
            'tgt_ids': torch.stack([item['tgt_ids'] for item in x]),
            'src_len': torch.tensor([item['src_len'] for item in x]),
            'tgt_len': torch.tensor([item['tgt_len'] for item in x]),
            'src_text': [item['src_text'] for item in x],
            'tgt_text': [item['tgt_text'] for item in x]
        }
    )
    
    # 初始化模型
    model = Seq2SeqModel(tokenizer.vocab_size, embed_size, tokenizer)
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 训练模型
    print("开始训练模型...")
    model, train_losses, valid_losses = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=n_epochs,
        save_dir=save_dir,
        patience=patience,
        clip=clip
    )
    
    
    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train loss')
    plt.plot(valid_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('train & validation loss curve')
    plt.savefig(os.path.join(save_dir, 'loss_history.png'))
    plt.show()
