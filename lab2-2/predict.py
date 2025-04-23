import torch
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import os
from typing import List, Dict, Any

from myTokenizer import Tokenizer
from Datasets import E2EDataset
from utils import load_checkpoint

def generate_predictions(model: torch.nn.Module, 
                        tokenizer: Tokenizer,
                        test_dataset: E2EDataset,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                        batch_size: int = 16,
                        max_length: int = 100,
                        beam_size: int = 3) -> List[str]:
    """
    使用模型生成测试集的预测结果
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        test_dataset: 测试数据集
        device: 设备 ('cuda' 或 'cpu')
        batch_size: 批量大小
        max_length: 生成的最大长度
        beam_size: 束搜索大小
        
    Returns:
        预测的文本列表
    """
    model.eval()
    model = model.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            src_ids = batch['src_ids'].to(device)
            
            # 假设模型有generate方法进行生成
            # 不同模型可能需要不同的生成代码
            if hasattr(model, 'generate'):
                # 使用模型的generate方法（如Transformer模型）
                generated_ids = model.generate(
                    input_ids=src_ids,
                    max_length=max_length,
                    num_beams=beam_size,
                    early_stopping=True
                )
            else:
                # 使用自定义的生成逻辑（如RNN模型）
                # 这里需要根据具体模型结构实现
                # 以下是示例代码，实际使用时需要替换
                batch_size = src_ids.size(0)
                decoder_input = torch.tensor([[tokenizer.BOS_ID]] * batch_size).to(device)
                generated_ids = []
                
                for _ in range(max_length):
                    # 前向传播
                    output = model(src_ids, decoder_input)
                    
                    # 获取下一个词的预测
                    next_word = output.argmax(-1)[:, -1].unsqueeze(1)
                    decoder_input = torch.cat([decoder_input, next_word], dim=-1)
                    
                    # 如果所有序列都生成了EOS标记，则停止
                    if (next_word == tokenizer.EOS_ID).all():
                        break
                
                generated_ids = decoder_input
            
            # 将生成的ID解码为文本
            for ids in generated_ids:
                text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                predictions.append(text)
    
    return predictions

def save_predictions(predictions: List[str], output_file: str) -> None:
    """
    将预测结果保存到CSV文件
    
    Args:
        predictions: 预测文本列表
        output_file: 输出文件路径
    """
    df = pd.DataFrame({'prediction': predictions})
    df.to_csv(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型生成预测')
    parser.add_argument('--test_file', type=str, required=True, help='测试数据CSV文件路径')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='分词器路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='输出预测结果的CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--max_length', type=int, default=100, help='生成的最大长度')
    parser.add_argument('--beam_size', type=int, default=3, help='束搜索大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 加载分词器
    tokenizer = Tokenizer.load(args.tokenizer_path)
    
    # 加载测试数据集
    test_dataset = E2EDataset(
        csv_file=args.test_file,
        tokenizer=tokenizer,
        is_test=True  # 标记为测试集
    )
    
    # 加载模型（这里需要根据你的实际模型结构进行修改）
    # 示例：假设你有一个名为Seq2SeqModel的模型
    from model import Seq2SeqModel  # 请替换为你的实际模型导入
    model = Seq2SeqModel(
        vocab_size=tokenizer.get_vocab_size(),
        # 其他必要的参数
    )
    
    # 加载模型状态
    checkpoint = load_checkpoint(model, path=args.model_path)
    
    # 生成预测
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        beam_size=args.beam_size
    )
    
    # 保存预测结果
    save_predictions(predictions, args.output_file)

if __name__ == "__main__":
    main()
