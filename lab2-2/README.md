# E2E 数据集 Transformer 文本生成任务

本项目实现了基于Transformer模型的文本生成任务，使用E2E数据集进行训练。

## 数据集准备

E2E (End-to-End) 数据集是一个用于自然语言生成任务的数据集，包含结构化意义表示 (MR) 和对应的自然语言描述。

### 1. 下载数据集

可以从以下链接下载E2E数据集：
https://github.com/tuetschek/e2e-dataset

将下载的数据放在 `data/raw` 目录下。

### 2. 预处理数据

使用以下命令预处理数据：

```bash
python scripts/preprocess_e2e.py --data_dir data/raw --output_dir data/processed
```

### 3. 构建词汇表

使用以下命令从数据集构建词汇表：

```bash
python scripts/build_vocab.py --data_dir data/processed --output_dir vocab --vocab_size 5000 --min_freq 2
```

## 模型训练

本项目包含以下主要组件：

1. `myTokenizer.py`: 实现了单词级别的分词器，用于处理英文文本
2. `myTransformer.py`: 实现了Encoder-Decoder架构的Transformer模型
3. `scripts/`: 包含数据处理和训练脚本

### 使用示例

使用预训练词汇表预处理数据：

```bash
python scripts/preprocess_e2e.py --data_dir data/raw --output_dir data/processed --vocab_file vocab/e2e_vocab.json
```

## 模型结构

项目实现了完整的Encoder-Decoder架构的Transformer模型，包括：

- 多头自注意力机制
- 位置编码
- 前馈神经网络
- 自回归文本生成

## 注意事项

- 推荐使用NLTK进行分词，确保已安装并下载相应的模型：
  ```python
  import nltk
  nltk.download('punkt')
  ```
- 对于大规模数据集，构建词汇表可能需要较长时间
