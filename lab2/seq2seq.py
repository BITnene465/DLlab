"""
seq2seq 模型的定义和训练函数的定义
"""
from torch import nn
import torch 
import torch.nn.functional as F
from myTokenizer import Tokenizer

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        
        self.temperature = nn.Parameter(torch.ones(1))  # 温度参数，用于控制注意力分布的平滑程度

    def forward(self, query, keys, valid_src_len=None):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))) / self.temperature
        scores = scores.squeeze(2).unsqueeze(1)

        # 如果有有效长度，则使用掩码来忽略填充部分
        if valid_src_len is not None:
            mask = self.get_mask(scores, valid_src_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

    def get_mask(self, scores, valid_src_len):
        """获得掩码，标记有效位置为1，填充位置为0"""
        batch_size, _, total_l = scores.size()
        mask = torch.zeros(batch_size, 1, total_l, device=scores.device)
        
        for i, l in enumerate(valid_src_len):
            mask[i, :, :l] = 1
        
        return mask
    

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p = 0.2, *args, **kwargs):
        """
        Args:
            embedding: 使用 nn.Embedding 创建的嵌入层
            hidden_size: 隐藏层的维度，也是词嵌入维度
            dropout_prob: Dropout 的概率，用于防止过拟合
        """
        super().__init__(*args, **kwargs)
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            batch_first=True, 
            num_layers=3, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(p = dropout_p)

        # 将双向LSTM的输出映射到单向
        self.fc = nn.Linear(hidden_size*2, hidden_size) 
        
        # 归一化层
        self.embed_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids, valid_src_len):
        """
        Args:
            input_ids: (batch_size, seq_length)
            valid_src_len: (batch_size) 每条token序列的有效信息长度
        
        Returns:
            output: (batch_size, seq_length, hidden_size)
            hidden: tuple[(num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size)] 同时包括了 h_n 和 c_n
        """
        # 没有传入就是全选
        # if valid_src_len is None:
        #     valid_src_len = torch.full((input_ids.size(0),), input_ids.size(1), dtype=torch.long) # 需要这个的 device 为 cpu
        
        embedded = self.embedding(input_ids)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)
        
        # 使用工具打包序列 -- 为了能够在一个 batch 中处理边长的序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            valid_src_len,
            batch_first=True,
            enforce_sorted=False,   # 不要求提前按照有效长度大小排序
        )
        packed_output, hidden = self.lstm(packed_embedded)
        
        # 使用工具解包
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        
        # 处理双向输出，形状为(batch_size, seq_len, hidden_size*2)
        output = self.fc(output)  # 将双向输出映射回(batch_size, seq_len, hidden_size)
        
        # 处理隐藏状态以适配解码器 -- 双向LSTM的隐藏状态形状为(num_layers*2, batch, hidden_size)
        num_layers = hidden[0].size(0) // 2
        
        # 重组隐藏状态 - 将前向和后向的隐状态合并
        # 提取前向和后向的隐藏状态，分别是奇数层和偶数层
        hidden_forward = hidden[0][:num_layers]
        hidden_backward = hidden[0][num_layers:]
        
        # 合并前向和后向隐藏状态
        hidden_combined = hidden_forward + hidden_backward
        
        # 做同样的处理对cell状态
        cell_forward = hidden[1][:num_layers]
        cell_backward = hidden[1][num_layers:]
        cell_combined = cell_forward + cell_backward   # todo 融合方式有待改进
       
        output = self.output_norm(output)   # 输出归一化
        
        return output, (hidden_combined, cell_combined) 
    

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.2, *args, **kwargs):
        """
        Args:
            embedding: 使用 nn.Embedding 创建的嵌入层
            hidden_size: 隐藏层的维度，也是词嵌入维度
            output_size: 输出层维度，也是 vocab_size
            dropout_prob: Dropout 的概率，用于防止过拟合
        """
        super().__init__(*args, **kwargs)
        
        self.embedding = nn.Embedding(output_size, hidden_size)  # 这里的 vocab_size 也是 decoder 的词表大小
        self.attention = BahdanauAttention(hidden_size=hidden_size)  # 注意力层
        self.lstm = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=3) # 此处 LSTM 的隐藏层结构需要和 encoder 的 LSTM 一致
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # 归一化层
        self.embed_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        
    def forward(self, encoder_outputs, encoder_hidden, valid_src_len, max_tgt_len=50, start_token_id=0, target=None):
        """
        使用 greedy search 或者 teacher forcing ，主要用于训练阶段
        Args:
            encoder_outputs : (batch_size, seq_length, hidden_size)
            encoder_hidden : (num_layers, batch_size, hidden_size)
            valid_src_len: (batch_size) 每条token序列的有效信息长度
            max_seq_length : 生成的最长序列长度 
            start_token_id : 用于decode的初始token对应的id
            target : 目标序列，用于 teacher forcing
        
        Return:
            decoder_outputs: (batch_size, seq_len, output_size) 
            decoder_hidden: 最后的 h_n, c_n 组成的元组
            attentions: (batch_size, max_tgt_len, src_l)
        """
        # 初始化 decoder_input -> (batch_size, 1) 全是 start_token_id
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=encoder_outputs.device)
        # 用 encoder 最后的隐藏层初始化 decoder 的隐藏层
        decoder_hidden = encoder_hidden 
        
        decoder_outputs = []  # 记录decoder输出词表id序列
        attentions = [] # 记录注意力矩阵序列，用于可视化
        
       
        for i in range(max_tgt_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                input_id = decoder_input,
                hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                valid_src_len=valid_src_len
            )
            
            decoder_outputs.append(decoder_output.unsqueeze(1))
            attentions.append(attn_weights)
            
            if target is not None:
                # teacher forcing
                decoder_input = target[:, i].unsqueeze(1)
            
            else:
                # 使用自己的预测结果作为输入 -- greedy search
                # decoder_output: (batch_size, output_size)
                _, topi = decoder_output.topk(1)
                decoder_input = topi.detach()   # 此处有必要detach,否则会导致该步的生成影响到前面所有步的梯度
                
            
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        
        return decoder_outputs, decoder_hidden, attentions
            
    
    
    def forward_step(self, input_id, hidden, encoder_outputs, valid_src_len=None):
        """
        Args:
            input_id: (batch_size, 1) 单个时间步输入(id值)
            hidden: tuple[(num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size)]  上一步结束时的隐藏状态(h_n 和 c_n)，用于初始化当前步的隐藏状态
            encoder_outputs : (batch_size, seq_length, hidden_size) 用于注意力查询
            valid_src_len: (batch_size) 每条token序列的有效信息长度
        
        Returns:
            output: (batch_size, output_size) 预测的概率分布
            hidden: 当前的步结束时的隐藏状态
        """
        embedded = self.embedding(input_id)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)
                
        # hidden: (h_n, c_n)，h_n:(num_layers, batch_size, hidden_size), 使用最后一层向量作为查询
        query = hidden[0][-1].unsqueeze(1)  
        
        context, attn_weights = self.attention(query, encoder_outputs, valid_src_len)
        context = self.context_norm(context)
        
        input_lstm = torch.cat((embedded, context), dim=2) 
        output, hidden = self.lstm(input_lstm, hidden)
        output = self.output_norm(output)   # 输出归一化
        output = self.fc_out(output.squeeze(1))
        
        return output, hidden, attn_weights
        
                
        
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, tokenizer: Tokenizer, dropout_p=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        
        self.encoder = Encoder(vocab_size, embed_size, dropout_p=dropout_p)
        self.decoder = AttnDecoder(embed_size, vocab_size, dropout_p=dropout_p)
        
        # 初始化模型参数
        self._init_parameters()
    
    
    def _init_parameters(self):
        """初始化模型参数"""
        print("开始初始化模型参数")
        # 注意力机制参数初始化
        for attention in [self.decoder.attention]:
            for name, param in attention.named_parameters():
                if 'weight' in name:
                    # 线性层权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    # 偏置初始化为0
                    nn.init.zeros_(param)
        
        # 初始化编码器新增的线性层
        nn.init.xavier_uniform_(self.encoder.fc.weight)
        nn.init.zeros_(self.encoder.fc.bias)
        
        # LSTM参数初始化
        for lstm in [self.encoder.lstm, self.decoder.lstm]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    # 输入权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    # 隐藏层权重使用正交初始化
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    # 偏置设为0，但是遗忘门偏置设为较大值以保留记忆
                    nn.init.zeros_(param)
                    if 'lstm' in name:
                        if lstm.bidirectional:
                            # 双向LSTM时，偏置形状会翻倍
                            param.data[lstm.hidden_size:2*lstm.hidden_size].fill_(1.0)  # 前向遗忘门偏置
                            if hasattr(lstm, 'bidirectional') and lstm.bidirectional:
                                param.data[3*lstm.hidden_size:4*lstm.hidden_size].fill_(1.0)  # 后向遗忘门偏置
                        else:
                            param.data[lstm.hidden_size:2*lstm.hidden_size].fill_(1.0)  # 遗忘门偏置
        
        # 输出层初始化
        nn.init.xavier_uniform_(self.decoder.fc_out.weight)
        nn.init.zeros_(self.decoder.fc_out.bias)
        
        # LayerNorm层通常使用默认初始化 
        print("seq2seq 模型初始化完毕")
    
    
    def forward(self, input_ids, valid_src_len, max_tgt_len, target_ids=None, teacher_forcing_ratio=1.0):
        
        # if add_eos_token_id:
        #     eos_token_id = self.tokenizer.eos_token_id
        #     eos_column = torch.full((input_ids.size(0), 1), eos_token_id, dtype=torch.long, device=input_ids.device)
        #     input_ids = torch.cat((input_ids, eos_column), dim=1)
    
        # input_ids 已经包含 eos_token, 并且已经被填充或者截断 （预处理阶段）
        encoder_outputs, encoder_hidden = self.encoder(input_ids, valid_src_len=valid_src_len)
        if target_ids is None: # 不使用 teacher forcing -- 使用 greedy search
            decoder_outputs, decoder_hidden, attentions = self.decoder(
                                                                    encoder_outputs=encoder_outputs, 
                                                                    encoder_hidden=encoder_hidden,
                                                                    valid_src_len = valid_src_len,
                                                                    max_tgt_len = max_tgt_len, 
                                                                    start_token_id=self.tokenizer.BOS_ID,
                                                                    target=None)
        else:   # 使用teacher forcing
            if torch.rand(1).item() >= teacher_forcing_ratio:
                target_ids = None   # 如果大于ratio，则不使用 teacher forcing
            decoder_outputs, decoder_hidden, attentions = self.decoder(
                                                                encoder_outputs=encoder_outputs, 
                                                                encoder_hidden=encoder_hidden,
                                                                valid_src_len = valid_src_len,
                                                                max_tgt_len = max_tgt_len, 
                                                                start_token_id=self.tokenizer.BOS_ID,
                                                                target=target_ids)

        return decoder_outputs, decoder_hidden, attentions
    
        
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f"模型已从 {path} 加载")
    
    
if __name__ == "__main__":
    # 简单测试
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    tokenizer = Tokenizer.load(path="tokenizers/e2e_tokenizer.json")
    print(f"词汇表大小: {tokenizer.get_vocab_size()}")
    embed_size = 300
    model = Seq2SeqModel(tokenizer.get_vocab_size(), embed_size, tokenizer)
    model.to(device)
    print("模型结构:")
    print(model)
    
    # 创建一个小批次的测试数据
    batch_size = 8
    src_seq_len = 30
    tgt_seq_len = 40
    
    # 随机生成输入序列 (假设 id 范围在 [0, vocab_size-1])
    input_ids = torch.randint(0, tokenizer.get_vocab_size(), (batch_size, src_seq_len), device=device)
    target_ids = torch.randint(0, tokenizer.get_vocab_size(), (batch_size, tgt_seq_len), device=device)
    valid_src_len = torch.randint(1, src_seq_len + 1, (batch_size,))   # 必须在cpu上
    
    print(f"输入形状: {input_ids.shape}")
    print(f"目标形状: {target_ids.shape}")
    
    # 尝试前向传播
    try:
        print("开始前向传播...")
        decoder_outputs, decoder_hidden, attentions = model(
            input_ids=input_ids,
            valid_src_len=valid_src_len,
            max_tgt_len=tgt_seq_len,
            target_ids=target_ids
        )
        
        print("前向传播成功!")
        print(f"解码器输出形状: {decoder_outputs.shape}")
        print(f"解码器隐藏状态形状: {decoder_hidden[0].shape}")
        print(f"注意力权重形状: {attentions.shape}")
        
        # 检查输出是否为预期形状
        expected_output_shape = (batch_size, tgt_seq_len, tokenizer.get_vocab_size())
        assert decoder_outputs.shape == expected_output_shape, f"输出形状不符合预期: {decoder_outputs.shape} vs {expected_output_shape}"
        
        # 检查解码器最后隐藏状态形状
        num_layers = 3  # 根据你的模型定义
        expected_hidden_shape = (num_layers, batch_size, embed_size)
        assert decoder_hidden[0].shape == expected_hidden_shape, f"隐藏状态形状不符合预期: {decoder_hidden[0].shape} vs {expected_hidden_shape}"
        
        # 检查注意力权重形状
        expected_attn_shape = (batch_size, tgt_seq_len, src_seq_len)
        assert attentions.shape == expected_attn_shape, f"注意力权重形状不符合预期: {attentions.shape} vs {expected_attn_shape}"
        
    except Exception as e:
        print(f"前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()

