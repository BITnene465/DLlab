import os
import pandas as pd
from myTokenizer import Tokenizer
from utils import reformat_mr


def get_corpus_from_dataset(dataset_dir: str):
    """
    从数据集目录中获取语料库if __name__ == "__main__":
    om_dataset(os.path.join(ROOT_DIR, "e2e_dataset"))
    Args:
        dataset_dir: 数据集所在目录路径
        b_capacity=5000, min_freq=1, lower=True)
    Returns:
        list: 包含所有参考文本(ref列)的列表e_dir, filename))
    """
    corpus = []
        # 遍历目录下的所有文件
    for filename in ["trainset.csv", "devset.csv", "testset.csv"]:
        file_path = os.path.join(dataset_dir, filename)
        df = pd.read_csv(file_path)
        # 提取'mr'列的内容
        if 'mr' in df.columns:  
            mr_texts = df['mr'].apply(reformat_mr).to_list()
            corpus.extend(mr_texts)
        else:
            print(f"{filename} 中没有找到'mr'列")
        # 提取'ref'列的内容
        if 'ref' in df.columns:
            ref_texts = df['ref'].tolist()
            corpus.extend(ref_texts)
        else:
            print(f"{filename} 中没有找到'ref'列")
    return corpus


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    corpus = get_corpus_from_dataset("e2e_dataset")
    save_dir = "tokenizers"
    filename = "e2e_tokenizer.json"
    tokenizer = Tokenizer(vocab_capacity=5000, min_freq=1, lower=True)
    tokenizer.build_vocab(corpus=corpus)
    tokenizer.save(path=os.path.join(save_dir, filename))