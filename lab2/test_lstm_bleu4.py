#! 在测试集上计算模型的 Bleu4 分数
from utils import calculate_bleu4
from myTokenizer import Tokenizer
from seq2seq import Seq2SeqModel
from train_lstm import validate

import torch
from torch.utils.data import DataLoader
from Datasets import E2EDataset
import os

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    max_src_len = 60
    max_tgt_len = 60
    tokenizer_path = "Tokenizers/e2e_tokenizer.json"
    model_path = "best.pt"
    testset_path = "e2e_dataset/testset_w_refs.csv"
    
    
    tokenizer = Tokenizer.load(tokenizer_path)
    model = Seq2SeqModel(vocab_size=tokenizer.get_vocab_size(), embed_size=256, tokenizer=tokenizer)
    model.load_model(model_path)
    model.to(device)
    
    test_dataset = E2EDataset(testset_path, tokenizer, max_src_len=max_src_len, max_tgt_len=max_tgt_len, mode='valid')
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: {
            'src_ids': torch.stack([item['src_ids'] for item in x]),
            'src_len': torch.tensor([item['src_len'] for item in x]),
            'src_text': [item['src_text'] for item in x],
            'tgt_text_list': [item['tgt_text_list'] for item in x],
            'tgt_tokens_list': [item['tgt_tokens_list'] for item in x],
        }
    )
    bleu4 = validate(model, test_dataloader, device, max_tgt_len=max_tgt_len)
    print(f"Testset Bleu4 score: {bleu4}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()