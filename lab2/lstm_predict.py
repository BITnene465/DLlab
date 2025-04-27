from torch.utils.data import DataLoader
import torch
import os

from myTokenizer import Tokenizer
from Datasets import E2EDataset
from seq2seq import Seq2SeqModel
from utils import mytqdm


def predict(test_loader, model, device, max_gen_len=50):
    """
    测试模型并生成注意力图
    
    Args:
        test_loader: 测试数据加载器
        model: 模型
        device: 计算设备
        max_gen_len: 生成序列的最大长度
    """ 
    tokenizer = model.tokenizer
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for batch in mytqdm(test_loader, desc="testing"):
            src_ids = batch['src_ids'].to(device)
            src_len = batch['src_len'].cpu()  
            src_text = batch['src_text']
            # 生成预测
            outputs, _, attentions = model(
                input_ids=src_ids,
                valid_src_len=src_len,
                max_tgt_len=max_gen_len,
                target_ids=None  # 不使用 teacher forcing
            )
            # 获取预测的token id
            predicted_ids = outputs.argmax(dim=2).cpu().numpy()
            generated_text = [tokenizer.decode(seq, skip_special_tokens=True, end_with_eos=True, clean_up_tokenization_spaces=True) for seq in predicted_ids]
            
            predictions.extend(generated_text)
   
    return predictions
           
def save_predictions_to_txt(save_path, predictions):
    with open(save_path, mode="w", encoding="utf-8") as f:
        for pre in predictions:
            f.write(pre + '\n')
            

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 参数设置
    vocab_path = "./tokenizers/e2e_tokenizer.json" 
    model_path = "./best.pt" 
    dataset_dir = "./e2e_dataset" 
    embed_size = 256  
    max_src_len = 60  
    batch_size = 128 
    max_gen_len = 50  
    save_path = "./predictions.txt"  
    device = 'cpu'
    
    # 设置工作目录和设备
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"使用设备: {device}")
    
    # 加载词汇表和模型``
    tokenizer = Tokenizer.load(path=vocab_path)
    model = Seq2SeqModel(
        vocab_size=tokenizer.get_vocab_size(), 
        embed_size=embed_size,
        tokenizer=tokenizer
    )
    
    # 加载预训练模型
    try:
        model.load_model(model_path)
        model.to(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit(1)
    
    # 加载测试数据集
    test_path = os.path.join(dataset_dir, "testset.csv")
    test_dataset = E2EDataset(
        test_path, 
        tokenizer, 
        max_src_len=max_src_len, 
        max_tgt_len=0,
        mode='test',
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,  
        shuffle=False, 
        collate_fn=lambda x: {
            'src_ids': torch.stack([item['src_ids'] for item in x]),
            'src_len': torch.tensor([item['src_len'] for item in x]),
            'src_text': [item['src_text'] for item in x],
        }
    )
    
    # 测试并保存
    predictions = predict(
        test_loader=test_loader, 
        model=model, 
        device=device, 
        max_gen_len=max_gen_len
    )
    save_predictions_to_txt(save_path=save_path, predictions=predictions)
    print(f"预测结果已保存到 {save_path}")