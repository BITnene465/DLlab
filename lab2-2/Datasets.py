# 数据集相关定义和逻辑
import re
import pandas as pd
from torch.utils.data import Dataset
import torch

from myTokenizer import Tokenizer


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


class E2EDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: Tokenizer, max_src_len: int = 50, max_tgt_len: int = 50, is_test: bool = False) -> None:
        """
        Args:
            csv_file: 数据文件路径
            tokenizer: 分词器对象
            max_src_len: 源序列最大长度
            max_tgt_len: 目标序列最大长度
            is_test: 是否为测试集（没有ref字段）
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8')  
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.is_test = is_test
        
        # 预处理数据 + 分词 + 转换为id序列
        self._process_data()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 直接从DataFrame获取预处理好的数据
        item = self.df.iloc[idx]
        result = {
            'src_ids': torch.tensor(item['src_ids']),
            'src_text': item['src_text'],
        }
        
        # 仅当非测试集时添加目标文本相关字段
        if not self.is_test and 'tgt_ids' in item and 'tgt_text' in item:
            result['tgt_ids'] = torch.tensor(item['tgt_ids'])
            result['tgt_text'] = item['tgt_text']
            
        return result
    
    def _process_data(self):
        # 1. 处理 mr 字段
        self.df['src_text'] = self.df['mr'].apply(reformat_mr)
        self.df['src_ids'] = self.df['src_text'].apply(
        lambda x: self.tokenizer.encode(x, add_special_tokens=True)
        )
        self.df['src_ids'] = self.tokenizer.pad_sequences(
            self.df['src_ids'].tolist(),
            max_length=self.max_src_len,
            padding=True,
            truncation=True
        )
    
        # 2. 处理目标文本(ref) - 仅当非测试集且存在ref字段时
        if not self.is_test:
            ref_field = 'ref'
            if ref_field:
                self.df['tgt_text'] = self.df[ref_field]
                self.df['tgt_ids'] = self.df['tgt_text'].apply(
                    lambda x: self.tokenizer.encode(x, add_special_tokens=True)
                )
                self.df['tgt_ids'] = self.tokenizer.pad_sequences(
                    self.df['tgt_ids'].tolist(),
                    max_length=self.max_tgt_len,
                    padding=True,
                    truncation=True
                )
            else:
                print("未找到参考文本字段，数据集将不包含目标文本")
    
    def get_grouped_data(self):
        """返回按MR分组的数据，用于多参考评估"""
        if self.is_test or 'tgt_text' not in self.df.columns:
            return None  # 测试集没有参考文本，无法分组
            
        grouped_data = {}
        for src_text, group in self.df.groupby('src_text'):
            first_item = group.iloc[0]
            grouped_data[src_text] = {
                'src_ids': torch.tensor(first_item['src_ids']),
                'tgt_text_list': group['tgt_text'].tolist(),
                'tgt_ids_list': [ids for ids in group['tgt_ids'].tolist()],
            }
        return grouped_data



