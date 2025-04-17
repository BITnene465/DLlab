# 小工具

import pandas as pd

def structure_mr_field(file_path):
    """将mr字段转换为结构化数据"""
    df = pd.read_csv(file_path, encoding='utf-8')
    for _, row in df.iterrows():
        mr_text = row.get('mr', None)
        if mr_text is not None:
            # 解析mr字段
            mr_dict = {}
            for item in mr_text.split(','):
                key, value = item.split('=')
                mr_dict[key.strip()] = value.strip()
            # 将结构化数据存回DataFrame
            df.loc[_, 'mr'] = str(mr_dict)
    return df