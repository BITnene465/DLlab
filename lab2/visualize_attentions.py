import os
import torch
from myTokenizer import Tokenizer
from seq2seq import Seq2SeqModel
from utils import plot_attention, reformat_mr

def visualize_attention(input_text, model_path, output_dir="./attention_plots", 
                        tokenizer_path="./tokenizers/e2e_tokenizer.json", 
                        embed_size=256, dropout_p=0.3):
    """
    可视化模型对输入文本的注意力
    
    Args:
        input_text: 输入文本
        model_path: 模型文件路径
        output_dir: 输出目录
        tokenizer_path: 分词器文件路径
        embed_size: 嵌入维度大小
        dropout_p: Dropout概率
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tokenizer = Tokenizer.load(tokenizer_path)
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqModel(tokenizer.get_vocab_size(), embed_size, tokenizer, dropout_p=dropout_p)
    model.load_model(model_path)
    model.to(device)
    model.eval()
   
    input_text = reformat_mr(input_text) # 格式化输入文本
    input_tokens = tokenizer.tokenize(input_text) # 分词
    input_tokens = ['<BOS>'] + input_tokens + ['<EOS>'] 
    
    # 对输入文本进行分词和编码
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    src_len = torch.tensor([len(input_ids)])
    input_ids = torch.tensor([input_ids], device=device)
    
    
    with torch.no_grad():
        outputs, _, attentions = model(
            input_ids=input_ids,
            valid_src_len=src_len,
            max_tgt_len=50,  
            target_ids=None, 
        )
        
        # 获取预测结果
        predicted_ids = outputs.argmax(dim=2).cpu().numpy()[0]
        predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids, skip_special_tokens=True, end_with_eos=True)
        predicted_tokens = ['<BOS>'] + predicted_tokens + ['<EOS>'] 
        
        # 可视化注意力
        attention_weights = attentions[0]  # 取第一个样本的注意力权重
        # 生成文件名
        filename = os.path.join(output_dir, f"attention_{len(os.listdir(output_dir)) + 1}.png")
        # 绘制注意力热力图
        plot_attention(
            attention=attention_weights[:len(predicted_tokens), :len(input_tokens)],
            source_tokens=input_tokens,
            target_tokens=predicted_tokens,
            title="Attention Visualization",
            filename=filename
        )
        
        print(f"输入文本: {input_text}")
        print(f"生成文本: {predicted_text}")
        print(f"注意力热力图已保存到: {filename}")
        


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = "best.pt"  
    # input_text = "name[Blue Spice], eatType[coffee shop], area[city centre]"
    input_text = "name[The Golden Palace], eatType[restaurant], food[Italian], priceRange[moderate], area[riverside], familyFriendly[yes], near[The Rice Boat]"
    
    visualize_attention(
        input_text=input_text, 
        model_path=model_path,
        output_dir="./attention_visualization"
    )