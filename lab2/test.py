from torch.utils.data import DataLoader
import torch
import os

from build_vocab import get_tokenizer_from_file
from datasets import E2EDataset
from seq2seq import Seq2SeqModel


def test(test_loader, model, save_path):
    pass


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 构建模型
    batch_size = 64
    tokenizer = get_tokenizer_from_file(vocab_path="vocab.json")
    model = Seq2SeqModel(
        vocab_size=tokenizer.vocab_size, 
        embed_size=300,
        tokenizer=tokenizer
        )
    model.load_model("./saved_models/model_epoch17_loss1.4159.pt")
    # 测试集
    dataset_dir = "./e2e_dataset"
    test_path = os.path.join(dataset_dir, "testset.csv") 
    test_dataset = E2EDataset(
        test_path, 
        tokenizer, 
        max_src_len=30, 
        max_tgt_len=0   # 为 0 表示不需要，target 相关数据
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
    # 简单测试
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            src_ids = batch['src_ids']
            src_len = batch['src_len']
            src_text = batch['src_text']
            
            # Generate predictions
            outputs, _, _ = model(
                input_ids=src_ids,
                valid_src_len=src_len,
                max_tgt_len=30,    # 生成序列的最大长度
                target_ids=None   # 非训练，不使用 teacher forcing
            )
            outputs = outputs.argmax(dim=2).cpu().numpy().tolist()
    
            # Display the input and output
            for i in range(len(src_text)):
                print(f"Input: {src_text[i]}")
                print(f"Output: {tokenizer.decode_engtext(outputs[i])}")
            break  # Only process one batch
    
    # 测试
    