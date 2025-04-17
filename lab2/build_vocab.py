import json
import os
import re
import nltk
from collections import Counter

from utils import structure_mr_field

# 词汇表类
class MyVocabulary:
    def __init__(self):
        self.idx2token = {}
        self.token2idx = {}
        self.special_tokens = {
            "<PAD>": 0,  
            "<UNK>": 1,  
            "<SOS>": 2,  
            "<EOS>": 3   
        }
        for token, idx in self.special_tokens.items():
            self.idx2token[idx] = token
            self.token2idx[token] = idx

    def add_word(self, word):
        if word not in self.token2idx:
            idx = len(self.idx2token)
            self.idx2token[idx] = word
            self.token2idx[word] = idx

    def build_vocab(self, words, min_freq=3):
        """
        Args:
            words (list[str]): 分词后所有的词汇放入一个列表中
            min_freq (int, optional): 最小词频，小于此值则不会加入词汇表
        """
        assert len(self.idx2token) == len(self.special_tokens), "词汇表应仅包含特殊标记"
        
        counter = Counter(words)
        for word, freq in counter.items():
            if freq >= min_freq:
                self.add_word(word)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'idx2token': self.idx2token, 'token2idx': self.token2idx, 'special_tokens': self.special_tokens}, f)
        print(f"词汇表存储在 {path}")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 将idx2token的键从字符串转回整数
            self.idx2token = {int(k): v for k, v in data['idx2token'].items()}
            self.token2idx = data['token2idx']
            self.special_tokens = data['special_tokens']
        print(f"读取 {path} 的词汇表成功")

# tokenizer 类
class MyTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        
        self.vocab_size = len(vocab.idx2token)
    
    @staticmethod
    def tokenize_eng(eng_text, lower_case=True):
        if lower_case:
            eng_text = eng_text.lower()
        eng_text = eng_text.strip()
        eng_tokens = nltk.tokenize.word_tokenize(eng_text)
        return eng_tokens


    def pad(self, tokens, max_l=50):
        """
        长于 max_l 就截断，短于 max_l 就填充, 并且保证每一句的有效信息最后都有 eos token
        Args:
            tokens list[str]: token 序列
        """
        if max_l == 0:   # 特判，不需要这个token序列的情况
            return []
        if len(tokens) > max_l-1:
            return tokens[:max_l-1] + ["<EOS>"]
        else:
            return tokens + ["<EOS>"] + ["<PAD>"] * (max_l - len(tokens) - 1)

    def encode(self, tokens):
        ids = [self.vocab.token2idx.get(token, self.vocab.special_tokens["<UNK>"]) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.vocab.idx2token.get(idx, "<UNK>") for idx in ids]
        return tokens
    
    def decode_engtext(self, ids, remove_special_tokens=True):
        tokens = self.decode(ids)
        if remove_special_tokens:
            tokens = [token for token in tokens if token not in self.vocab.special_tokens]
        return ' '.join(tokens)
    
    @property
    def sos_token_id(self):
        try:
            return self.vocab.token2idx["<SOS>"]
        except KeyError:
            raise KeyError("The <SOS> token is not found in the vocabulary.")

    @property 
    def eos_token_id(self):
        try:
            return self.vocab.token2idx["<EOS>"]
        except KeyError:
            raise KeyError("The <EOS> token is not found in the vocabulary.")
    
    @property
    def pad_token_id(self):
        try:
            return self.vocab.token2idx["<PAD>"]
        except KeyError:
            raise KeyError("The <PAD> token is not found in the vocabulary.")
        
    @property
    def unk_token_id(self):
        try:
            return self.vocab.token2idx["<UNK>"]
        except KeyError:
            raise KeyError("The <UNK> token is not found in the vocabulary.")


# 
def get_tokenizer_from_file(vocab_path):
    vocab = MyVocabulary()
    vocab.load(vocab_path)
    tokenizer = MyTokenizer(vocab=vocab)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    return tokenizer

#! 脚本部分：利用数据集构建词汇表
def tokenize_csv_file(file_path):
    """从CSV文件中读取并标记文本"""
    all_tokens = []
    df = structure_mr_field(file_path)
    for _, row in df.iterrows():
        # 处理mr字段（输入）
        mr_text = row.get('mr', None)
        if mr_text is not None:
            mr_tokens = MyTokenizer.tokenize_eng(mr_text)
            all_tokens.extend(mr_tokens)
        
        # 处理ref字段（输出）
        ref_text = row.get('ref', None)
        if ref_text is not None:
            ref_tokens = MyTokenizer.tokenize_eng(mr_text)
            all_tokens.extend(ref_tokens)
    
    return all_tokens

def build_vocabulary_from_datasets(train_path, valid_path, test_path, output_path, min_freq=2):
    """从所有数据集构建词汇表"""
    print(f"从训练集 {train_path} 中提取词汇...")
    train_tokens = tokenize_csv_file(train_path)
    print(f"从验证集 {valid_path} 中提取词汇...")
    valid_tokens = tokenize_csv_file(valid_path)
    print(f"从测试集 {test_path} 中提取词汇...")
    test_tokens = tokenize_csv_file(test_path)
    
    # 合并所有标记
    all_tokens = train_tokens + valid_tokens + test_tokens
    print(f"总共收集到 {len(all_tokens)} 个标记")
    
    # 构建词汇表
    vocab = MyVocabulary()
    vocab.build_vocab(all_tokens, min_freq=min_freq)
    print(f"按最小频率 {min_freq} 构建的词汇表大小: {len(vocab.token2idx)}")
    
    # 存储词汇表
    vocab.save(output_path)
    
    return vocab


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    dataset_dir = "./e2e_dataset"
    train_path = os.path.join(dataset_dir, "trainset.csv")
    valid_path = os.path.join(dataset_dir, "devset.csv") 
    test_path = os.path.join(dataset_dir, "testset.csv")
    
    output_path = "./vocab.json"
    
    # 构建词汇表
    vocab = build_vocabulary_from_datasets(train_path, valid_path, test_path, output_path, min_freq=1)
    
    # 打印词汇表统计信息
    print(f"词汇表大小: {len(vocab.token2idx)}")
    print(f"特殊标记: {vocab.special_tokens}")
    print(f"前10个词: {list(vocab.token2idx.items())[:10]}")

