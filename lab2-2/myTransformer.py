import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 注册为非参数buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入嵌入 [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        q = self.wq(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.wo(context)
        return output, attention

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return x

class Decoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x

class Transformer(nn.Module):
    def __init__(self, tokenizer, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.output_linear(dec_output)
        return output
    
    def encode(self, src, src_mask):
        """编码源序列"""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """解码目标序列"""
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)
    
    def generate(self, src, src_mask, max_len, start_symbol_id, end_symbol_id=None, temperature=1.0):
        """自回归生成文本序列"""
        # 编码源序列
        enc_output = self.encode(src, src_mask)
        
        # 初始化生成序列，起始为开始符号
        batch_size = src.shape[0]
        ys = torch.ones(batch_size, 1).fill_(start_symbol_id).long().to(src.device)
        
        for i in range(max_len-1):
            # 创建目标掩码（下三角）
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
            
            # 解码当前序列
            out = self.decode(ys, enc_output, src_mask, tgt_mask)
            # 获取下一个词的概率
            prob = self.output_linear(out[:, -1])
            prob = F.softmax(prob / temperature, dim=-1)
            
            # 采样下一个词
            next_word = torch.multinomial(prob, num_samples=1)
            
            # 添加到生成序列
            ys = torch.cat([ys, next_word], dim=1)
            
            # 如果生成了结束符号则停止
            if end_symbol_id is not None and (next_word == end_symbol_id).all():
                break
                
        return ys
    
    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        """加载模型参数"""
        self.load_state_dict(torch.load(path))
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        """生成下三角掩码（用于解码器自注意力）"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_padding_mask(seq: list[int], pad_idx=0):
    """创建padding掩码（用于屏蔽<PAD>标记）"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2).float()
    return mask
