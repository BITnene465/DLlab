import re
import os
import json
from collections import Counter
from nltk.tokenize import word_tokenize
from typing import List, Dict, Union, Tuple, Optional, Any


class Tokenizer:
    """
    分词器，处理特殊标记（PAD, UNK, BOS, EOS）
    """
    
    # 特殊标记
    PAD_ID = 0  # 用于填充
    UNK_ID = 1  # 未知词
    BOS_ID = 2  # 句子开始
    EOS_ID = 3  # 句子结束
    
    def __init__(self, 
                 vocab_capacity: int = None,
                 min_freq: int = 1,
                 lower: bool = True,
                 special_tokens: Dict[str, int] = None) -> None:
        """
        Args:
            vocab_size: 词汇表大小限制（包括特殊标记）
            min_freq: 词频最小阈值
            lower: 是否将文本转为小写
            special_tokens: 自定义特殊标记字典 {token_name: token_id}
        """
        # 基础设置
        self.vocab_capacity = vocab_capacity
        self.min_freq = min_freq
        self.lower = lower
        
        # 词汇表相关
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.num_words = 0
        
        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = {
                "<PAD>": self.PAD_ID,
                "<UNK>": self.UNK_ID, 
                "<BOS>": self.BOS_ID,
                "<EOS>": self.EOS_ID
            }
        
        # 将特殊标记添加到词汇表
        for token, idx in self.special_tokens.items():
            self.word2idx[token] = idx
            self.idx2word[idx] = token
        
        # 初始计数
        self.num_words = len(self.special_tokens)
        
    def preprocess(self, text: str) -> str:
        """
        文本预处理
        Args:
            text: 输入文本
        """
        if self.lower:
            text = text.lower()
        
        # 处理标点符号 - 在标点符号两边添加空格，使其成为单独的标记
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        将文本转换为 tokens (使用nltk)
        Args:
            text: 输入文本
            
        Returns:
            单词列表
        """
        text = self.preprocess(text)
        return word_tokenize(text=text)  # 使用 nltk 库进行分词
    
    def build_vocab(self, corpus: List[str], verbose: bool = True) -> None:
        """
        从语料库构建词汇表
        
        Args:
            corpus: 文本语料库
            verbose: 是否显示进度信息
        """
        if verbose:
            print("Building vocabulary...")
            
        # 计算所有单词的出现频率
        for text in corpus:
            words = self.tokenize(text)
            self.word_counts.update(words)
        
        # 按词频降序排列
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 添加符合条件的词到词汇表
        for word, count in sorted_words:
            if count < self.min_freq:
                continue
                
            # 如果达到词汇表大小限制，停止添加
            if self.vocab_capacity and self.num_words >= self.vocab_capacity:
                break
                
            if word not in self.word2idx:
                self.word2idx[word] = self.num_words
                self.idx2word[self.num_words] = word
                self.num_words += 1
                
        if verbose:
            print(f"Vocabulary built with {self.num_words} words.")
            
    def encode(self, text: Union[str, List[str]], 
               add_special_tokens: bool = False) -> List[int]:
        """
        将文本编码为ID序列
        
        Args:
            text: 输入文本或已分词的单词列表
            add_special_tokens: 是否添加BOS/EOS标记
            
        Returns:
            ID序列
        """
        # 如果输入是字符串，先进行分词
        if isinstance(text, str):
            tokens = self.tokenize(text)
        else:
            tokens = text
        
        # 添加特殊标记
        if add_special_tokens:
            tokens = ["<BOS>"] + tokens + ["<EOS>"]
            
        # 将标记转换为ID
        ids = []
        for token in tokens:
            if token in self.word2idx:
                ids.append(self.word2idx[token])
            else:
                ids.append(self.UNK_ID)
                
        return ids
    
    def batch_encode(self, texts: List[str], 
                    add_special_tokens: bool = False,
                    max_length: int = None,
                    padding: bool = False,
                    truncation: bool = False) -> List[List[int]]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            add_special_tokens: 是否添加BOS/EOS标记
            max_length: 最大序列长度
            padding: 是否进行填充
            truncation: 是否进行截断
            
        Returns:
            ID序列列表
        """
        encoded_texts = [self.encode(text, add_special_tokens) for text in texts]
        
        # 如果需要填充或截断
        if padding or truncation:
            encoded_texts = self.pad_sequences(
                encoded_texts, 
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
        
        return encoded_texts
    
    def decode(self, ids: List[int], 
               skip_special_tokens: bool = True,
               end_with_eos: bool = True,
               clean_up_tokenization_spaces: bool = True) -> str:
        """
        将ID序列解码为文本
        
        Args:
            ids: ID序列
            skip_special_tokens: 是否跳过特殊标记
            clean_up_tokenization_spaces: 是否清理分词过程中添加的多余空格
            
        Returns:
            解码后的文本
        """
        tokens = self.convert_ids_to_tokens(
            ids,
            skip_special_tokens=skip_special_tokens,
            end_with_eos=end_with_eos
        )
                
        text = " ".join(tokens)
        
        # 清理分词过程中添加的空格
        if clean_up_tokenization_spaces:
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            
        return text
    
    def convert_ids_to_tokens(self, ids: List[int],
                              skip_special_tokens: bool = True,
                              end_with_eos: bool = True) -> List[str]:
        """
        将ID序列转换为单词列表
        
        Args:
            ids: ID序列
            skip_special_tokens: 是否跳过特殊标记
            
        Returns:
            单词列表
        """
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        tokens = []
        for idx in ids:
            if end_with_eos and idx == self.EOS_ID:
                break
            if idx in special_ids:
                continue
            if idx in self.idx2word:
                tokens.append(self.idx2word[idx])
            else:
                tokens.append("<UNK>")
                
        return tokens
    
    def pad_sequences(self, sequences: List[List[int]], 
                     max_length: int = None,
                     padding: bool = True,
                     truncation: bool = False) -> List[List[int]]:
        """
        填充/截断序列到统一长度, 含batch维度
        
        Args:
            sequences: ID序列列表
            max_length: 最大长度，如果为None则使用最长序列长度
            padding: 是否填充短序列
            truncation: 是否截断长序列
            
        Returns:
            填充/截断后的序列列表
        """
        # 确定最大长度
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
            
        result = []
        for seq in sequences:
            # 创建序列的副本以避免修改原始数据
            seq_copy = seq.copy()
            # 截断
            if truncation and len(seq_copy) > max_length:
                seq_copy = seq_copy[:max_length]
            # 填充
            if padding and len(seq_copy) < max_length:
                seq_copy = seq_copy + [self.PAD_ID] * (max_length - len(seq_copy))
                
            result.append(seq_copy)
            
        return result
    
    def get_vocab(self) -> Dict[str, int]:
        """获取完整词汇表"""
        return self.word2idx
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.num_words
    
    def save(self, path: str) -> None:
        """       
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 确保idx2word的键保存为字符串但代表整数
        idx2word_str = {str(k): v for k, v in self.idx2word.items()}
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
            'config': {
                'vocab_capacity': self.vocab_capacity,
                'min_freq': self.min_freq,
                'lower': self.lower,
                'special_tokens': self.special_tokens,
                'num_words': self.num_words
            },
            'word2idx': self.word2idx,
            'idx2word': idx2word_str,
            'word_counts': dict(self.word_counts)
            }, f, ensure_ascii=False, indent=4)
            
        print(f"Tokenizer saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """
        加载已保存的分词器
        
        Args:
            path: 模型路径
            
        Returns:
            加载的分词器实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokenizer = cls(
            vocab_capacity=data['config']['vocab_capacity'],
            min_freq=data['config']['min_freq'],
            lower=data['config']['lower'],
            special_tokens=data['config']['special_tokens']
        )
        
        # 加载词汇表和计数
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        tokenizer.word_counts = Counter(data['word_counts'])
        tokenizer.num_words = data['config']['num_words']
        
        print(f"Tokenize loaded from {path}")
        return tokenizer