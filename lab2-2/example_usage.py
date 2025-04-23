import torch
from myTransformer import Transformer, create_padding_mask

# 定义超参数
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
max_seq_len = 100
dropout = 0.1

# 特殊标记
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

# 初始化模型
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
    dropout=dropout
)

# 创建一个示例输入
src = torch.randint(3, src_vocab_size, (2, 10))  # 批量大小为2，序列长度为10
tgt = torch.randint(3, tgt_vocab_size, (2, 8))   # 批量大小为2，序列长度为8

# 创建掩码
src_padding_mask = create_padding_mask(src)
tgt_padding_mask = create_padding_mask(tgt)
tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(src.device)

# 前向传播
output = model(src, tgt, src_padding_mask, tgt_mask)
print("Output shape:", output.shape)  # 应该是[2, 8, tgt_vocab_size]

# 示例文本生成
print("\n自回归生成示例:")
generated = model.generate(
    src=src,
    src_mask=src_padding_mask,
    max_len=20,
    start_symbol=BOS_IDX,
    end_symbol=EOS_IDX,
    temperature=0.7
)
print("Generated sequence shape:", generated.shape)
print("Generated sequences:")
print(generated)
