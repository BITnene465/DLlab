import os
import argparse
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def calculate_accuracy(df):
    """计算预测准确率"""
    if 'prediction' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Excel 文件必须包含 'prediction' 和 'answer' 列")
    # 检查是否有缺失值
    if df['prediction'].isna().any() or df['answer'].isna().any():
        print("警告: 存在缺失值，这些行将被视为预测错误")
    # 将两列转为字符串并去除前后空格
    df['prediction'] = df['prediction'].astype(str).str.strip()
    df['answer'] = df['answer'].astype(str).str.strip()
    # 计算匹配数
    correct = (df['prediction'] == df['answer']).sum()
    total = len(df)
    return correct, total, correct / total


def calculate_metrics(df):
    """计算多个评价指标：准确率、精确率、召回率和F1分数"""
    if 'prediction' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Excel 文件必须包含 'prediction' 和 'answer' 列")
    # 将两列转为字符串并去除前后空格
    df['prediction'] = df['prediction'].astype(str).str.strip()
    df['answer'] = df['answer'].astype(str).str.strip()
    # 获取所有可能的类别
    all_classes = sorted(set(df['answer'].unique()).union(set(df['prediction'].unique())))
    # 计算各项指标
    y_true = df['answer'].values
    y_pred = df['prediction'].values
    
    accuracy = accuracy_score(y_true, y_pred)
    # 对于precision, recall和f1，我们需要处理可能的警告
    try:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=all_classes)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=all_classes)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=all_classes)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        precision = recall = f1 = float('nan')
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classes': all_classes
    }


def calculate_category_metrics(df):
    """按类别计算评价指标"""
    category_stats = {}
    # 检查是否有类别列
    if 'category' in df.columns:
        categories = df['category'].unique()
        
        for category in categories:
            category_df = df[df['category'] == category]
            metrics = calculate_metrics(category_df)
            
            correct = (category_df['prediction'] == category_df['answer']).sum()
            total = len(category_df)
            
            category_stats[category] = {
                'correct': int(correct),
                'total': total,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
    return category_stats


def main():
    parser = argparse.ArgumentParser(description='计算预测准确率')
    parser.add_argument('--input', type=str, required=True, help='包含预测和答案的Excel文件路径')
    parser.add_argument('--sheet', type=str, default=0, help='Excel工作表名称或索引')
    
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"文件 {args.input} 不存在")
    # 读取Excel文件
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    # 计算综合评价指标
    metrics = calculate_metrics(df)
    # 打印总体结果
    print(f"总体评价指标:")
    print(f"  准确率(Accuracy): {metrics['accuracy']:.4f}")
    print(f"  精确率(Precision): {metrics['precision']:.4f}")
    print(f"  召回率(Recall): {metrics['recall']:.4f}")
    print(f"  F1分数(F1 Score): {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
