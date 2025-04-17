# 数据集相关定义和逻辑
import re
import pandas as pd
from torch.utils.data import Dataset
from build_vocab import MyTokenizer
import torch

class E2EDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_src_len=50, max_tgt_len=50):
        """
        Args:
            csv_file: 数据文件路径
            tokenizer: 分词器对象
            max_src_len: 源序列最大长度
            max_tgt_len: 目标序列最大长度
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8')  
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # 预处理所有数据
        self.processed_data = []
        for idx in range(len(self.df)):
            self.processed_data.append(self._process_row(idx))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 返回预处理好的数据
        return self.processed_data[idx]
    
    def get_grouped_data(self):
        """返回按MR分组的数据，用于多参考评估"""
        grouped_data = {}
        for item in self.processed_data:
            src_text = item['src_text']
            if src_text not in grouped_data:
                grouped_data[src_text] = {
                    'src_ids': item['src_ids'],
                    'src_len': item['src_len'],
                    'tgt_text_list': [item['tgt_text']],
                    'tgt_tokens_list': [self.tokenizer.decode(item['tgt_ids'].tolist())],
                }
            else:
                grouped_data[src_text]['tgt_text_list'].append(item['tgt_text'])
                grouped_data[src_text]['tgt_tokens_list'].append(self.tokenizer.decode(item['tgt_ids'].tolist()))
        return grouped_data
    
    
    def _process_row(self, idx):
        """预处理单行数据"""
        # 获取一条数据
        row = self.df.iloc[idx]
        # 预处理和分词源文本 (mr字段)
        src_text = row.get('mr', '') 
        src_tokens = MyTokenizer.tokenize_eng(src_text)
        # 预处理和分词目标文本 (ref字段)
        tgt_text = row.get('ref', '')
        tgt_tokens = MyTokenizer.tokenize_eng(tgt_text)
        # 填充或截断序列
        src_tokens_padded = self.tokenizer.pad(src_tokens, self.max_src_len)
        tgt_tokens_padded = self.tokenizer.pad(tgt_tokens, self.max_tgt_len)
        # 转换为ID
        src_ids = self.tokenizer.encode(src_tokens_padded)
        tgt_ids = self.tokenizer.encode(tgt_tokens_padded)
        # 计算有效长度（不包括填充）
        src_len = min(len(src_tokens) + 1, self.max_src_len)  # +1 是为了包括 EOS 标记
        tgt_len = min(len(tgt_tokens) + 1, self.max_tgt_len)  # +1 是为了包括 EOS 标记
        
        return {
            'src_ids': torch.tensor(src_ids),
            'tgt_ids': torch.tensor(tgt_ids),
            'src_len': src_len,
            'tgt_len': tgt_len,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
