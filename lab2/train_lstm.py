import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json

from seq2seq import Seq2SeqModel
from Datasets import E2EDataset
from myTokenizer import Tokenizer
from utils import plot_train_curve, calculate_bleu4, mytqdm

    
def validate(model: Seq2SeqModel, valid_dataloader, device, max_tgt_len=60):
    """使用多参考评估模型"""
    model.eval()
    tokenizer = model.tokenizer
    
    # 用于计算BLEU的参考和假设
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in mytqdm(valid_dataloader, desc="validating..."):
            src_ids = batch['src_ids'].to(device) 
            src_len = batch['src_len'].cpu()
            outputs, _, _ = model(
                input_ids=src_ids,
                valid_src_len=src_len,
                max_tgt_len=max_tgt_len,
                target_ids=None,  # 不使用 teacher forcing
            )
            predicted_ids = outputs.argmax(dim=2).cpu().numpy().tolist()
            predicted_tokens = [tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=True, end_with_eos=True) for seq in predicted_ids ]  # 解码预测的token
            
            # 将参考和预测添加到评估列表
            tgt_tokens_list = batch['tgt_tokens_list']
            references.extend(tgt_tokens_list)
            hypotheses.extend(predicted_tokens)
           
    # 计算BLEU-4
    bleu4 = calculate_bleu4(references, hypotheses)
    return bleu4


def train_one_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=1.0, clip=1.0):
    """
    Args:
        model: 模型实例
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        clip: 梯度裁剪阈值
        teacher_forcing_ratio: teacher forcing 比例
    
    Returns:
        epoch_loss: 平均损失
    """
    model.train()
    epoch_loss = 0
    
    for batch in mytqdm(dataloader, desc=f"training...  tf_ratio:{teacher_forcing_ratio:.2f}"):
        # 准备数据
        src_ids = batch['src_ids'].to(device)
        tgt_ids = batch['tgt_ids'][:, 1:].to(device) # 去掉第一个 BOS token  
        src_len = batch['src_len'].cpu()  # pack_padded_sequence 要求有效长度必须在 CPU 上
        # 训练
        optimizer.zero_grad()     
        output, _, _ = model(
            input_ids=src_ids,
            valid_src_len=src_len,
            max_tgt_len=tgt_ids.size(1),
            target_ids=tgt_ids,
            teacher_forcing_ratio=teacher_forcing_ratio,
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


def train(model, train_dataloader, valid_dataloader, optimizer, criterion, device, 
          n_epochs, save_dir, patience=3, clip=1.0, tf_ratio=1.0, tf_decay=0.0, tf_min=1.0, 
          best_model_path=None):
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
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, tf_ratio, clip)
        train_losses.append(train_loss)
        # 在验证集上评估
        valid_bleu = validate(model, valid_dataloader, device)
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
        
        # scheduled sampling 实现
        tf_ratio = max(tf_ratio-tf_decay, tf_min)
    
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


def get_default_config():
    """返回默认配置"""
    return {
        "vocab_path": "./tokenizers/e2e_tokenizer.json",
        "embed_size": 256,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "clip": 5.0,
        "n_epochs": 30,
        "patience": 10,
        "dropout_p": 0.3,
        "max_src_len": 40,
        "max_tgt_len": 40,
        "dataset_dir": "./e2e_dataset",
        "save_dir": None, # 使用超参数命名
        "best_model_path": None,  # 如果要继续训练，可以指定
        "tf_ratio": 1.0,  # teacher forcing ratio 初始值
        "tf_decay": 0.0,  # teacher forcing 衰减率
        "tf_min": 1.0,    # teacher forcing 最小值
    }

if __name__ == "__main__":
    import argparse
    import json
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='训练Seq2Seq模型')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, help='配置文件路径（JSON格式）')
    
    # 数据集参数
    parser.add_argument('--dataset_dir', type=str, help='数据集目录')
    parser.add_argument('--vocab_path', type=str, help='词汇表路径')
    parser.add_argument('--max_src_len', type=int, help='源序列最大长度')
    parser.add_argument('--max_tgt_len', type=int, help='目标序列最大长度')
    
    # 模型参数
    parser.add_argument('--embed_size', type=int, help='嵌入层大小')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--weight_decay', type=float, help='权重衰减')
    parser.add_argument('--clip', type=float, help='梯度裁剪阈值')
    parser.add_argument('--n_epochs', type=int, help='训练轮数')
    parser.add_argument('--patience', type=int, help='早停耐心值')
    parser.add_argument('--dropout_p', type=float, help='dropout probability')
    parser.add_argument('--tf_ratio', type=float, help='teacher forcing ratio (initial value)')
    parser.add_argument('--tf_decay', type=float, help='teacher forcing decay rate')
    parser.add_argument('--tf_min', type=float, help='minimum teacher forcing ratio')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, help='模型保存目录')
    parser.add_argument('--best_model_path', type=str, help='最佳模型路径（用于继续训练）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 首先获取默认配置
    config = get_default_config()
    
    # 如果提供了配置文件，从中加载配置覆盖默认值
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
                print(f"已从{args.config}加载配置")
        except Exception as e:
            print(f"读取配置文件出错: {e}")
            exit(1)
    
    # 命令行参数覆盖配置文件和默认值
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    
    # 显示最终配置
    print("训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    # 数据集路径
    dataset_dir = config["dataset_dir"]
    train_path = os.path.join(dataset_dir, "trainset.csv")
    valid_path = os.path.join(dataset_dir, "devset.csv")
    
    # 超参数
    vocab_path = config["vocab_path"]
    embed_size = config["embed_size"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    clip = config["clip"]
    n_epochs = config["n_epochs"]
    patience = config["patience"]
    dropout_p = config["dropout_p"]
    # teacher forcing 相关超参数
    tf_ratio = config["tf_ratio"]
    tf_decay = config["tf_decay"]
    tf_min = config["tf_min"]
    # 数据集超参数
    max_src_len = config["max_src_len"]
    max_tgt_len = config["max_tgt_len"]
    
    if config["save_dir"] is None:
        save_dir = f"./train_history/embed{embed_size}_batch{batch_size}_lr{learning_rate}_clip{clip}_dp{dropout_p}_tfdecay{tf_decay}_tfmin{tf_min}"    # 使用超参数命名
    else:
        save_dir = config["save_dir"]
    
    best_model_path = config["best_model_path"]
    
    # 加载词汇表
    tokenizer = Tokenizer.load(path=vocab_path)
    # 创建数据集
    train_dataset = E2EDataset(train_path, tokenizer, max_src_len=max_src_len, max_tgt_len=max_tgt_len, mode='train')
    valid_dataset = E2EDataset(valid_path, tokenizer, max_src_len=max_src_len, max_tgt_len=max_tgt_len, mode='valid')
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: {
            'src_ids': torch.stack([item['src_ids'] for item in x]),
            'tgt_ids': torch.stack([item['tgt_ids'] for item in x]),
            'src_len': torch.tensor([item['src_len'] for item in x]),
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
            'src_len': torch.tensor([item['src_len'] for item in x]),
            'src_text': [item['src_text'] for item in x],
            'tgt_text_list': [item['tgt_text_list'] for item in x],
            'tgt_tokens_list': [item['tgt_tokens_list'] for item in x],
        }
    )
    
    # 训练
    model = Seq2SeqModel(tokenizer.get_vocab_size(), embed_size, tokenizer, dropout_p=dropout_p)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID)  # 忽略 PAD token 的损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("开始训练模型...")
    model, train_losses, valid_bleus = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=n_epochs,
        save_dir=save_dir,
        patience=patience,
        clip=clip,
        tf_ratio=tf_ratio,
        tf_decay=tf_decay,
        tf_min=tf_min,
        best_model_path=best_model_path,
    )
    # 保存最终配置到 save_dir
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"配置已保存到 {config_path}")
    # 绘图
    plot_train_curve(train_losses, valid_bleus, save_dir)