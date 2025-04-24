# 数据集相关定义和逻辑
import pandas as pd
from torch.utils.data import Dataset
import torch

from utils import reformat_mr

class E2EDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_src_len=50, max_tgt_len=50, mode='train'):
        """
        Args:
            csv_file: 数据文件路径
            tokenizer: 分词器对象
            max_src_len: 源序列最大长度
            max_tgt_len: 目标序列最大长度
        """
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.mode = mode
        
        # 预处理数据
        df = pd.read_csv(csv_file, encoding='utf-8')  
        self.df = self._preprocess_data(df)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 直接从DataFrame获取预处理好的数据, mode不同，返回结构不同
        item = self.df.iloc[idx]
        if self.mode == 'train':
            return {
                'src_text': item['src_text'],
                'src_len': torch.tensor(item['src_len']),
                'src_ids': torch.tensor(item['src_ids']),
                'tgt_text': item['tgt_text'],
                'tgt_ids': torch.tensor(item['tgt_ids']),
            }
        elif self.mode == 'valid':
            return {
                'src_text': item['src_text'],
                'src_len': torch.tensor(item['src_len']),
                'src_ids': torch.tensor(item['src_ids']),
                'tgt_text_list': item['tgt_text_list'],
                'tgt_tokens_list': item['tgt_tokens_list'],
            }
        elif self.mode == 'test':
            return {
                'src_text': item['src_text'],
                'src_len': torch.tensor(item['src_len']),
                'src_ids': torch.tensor(item['src_ids']),
            }
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'valid', or 'test'.")
        
    def _preprocess_data(self, df):
        # 验证集特殊处理
        if self.mode == 'valid':
            # 按照 MR分组
            group_data = df.groupby('mr')
            df = group_data.agg({
                'ref': lambda x: list(x)
            }).reset_index()
            
            df = df.rename(columns={'mr': 'src_text'})
            df['src_text'] = df['src_text'].apply(lambda x: reformat_mr(x))
            df['src_ids'] = df['src_text'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
            df['src_len'] = df['src_ids'].apply(lambda x: min(len(x), self.max_src_len))
            df['src_ids'] = self.tokenizer.pad_sequences(
                df['src_ids'].tolist(), max_length=self.max_src_len, padding=True, truncation=True
            )
            
            df = df.rename(columns={'ref': 'tgt_text_list'})
            df['tgt_tokens_list'] = df['tgt_text_list'].apply(lambda x: [self.tokenizer.tokenize(text) for text in x])
            
            return df
            
        # 非验证集情况
        df = df.rename(columns={'mr': 'src_text'})
        df['src_text'] = df['src_text'].apply(lambda x: reformat_mr(x))
        df['src_ids'] = df['src_text'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
        df['src_len'] = df['src_ids'].apply(lambda x: min(len(x), self.max_src_len))
        df['src_ids'] = self.tokenizer.pad_sequences(
            df['src_ids'].tolist(), max_length=self.max_src_len, padding=True, truncation=True
        )
        
        
        if self.mode == 'train':
            df = df.rename(columns={'ref': 'tgt_text'})
            df['tgt_ids'] = self.tokenizer.batch_encode(
                df['tgt_text'].tolist(),
                add_special_tokens=True,
                max_length=self.max_tgt_len,
                padding=True,
                truncation=True
            )
        return df
    
