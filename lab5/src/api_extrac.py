import os
import time
import re
from openai import OpenAI

# 1. 配置区域 
STU_NAME = "tjy"
# 通用API Key
GENERAL_API_KEY = "Nothing"
# 输入新闻文件的文件夹路径
INPUT_NEWS_DIR = "./news_data" 
# 输出结果的文件夹路径
OUTPUT_DIR = "./extraction"
# 提示词配置文件的路径
PROMPT_DIR = "./prompts"

"""
每个模型是一个字典，包含
- 'name': 用于文件命名的模型简称 (不要有特殊字符)
- 'api_model_name': 调用API时使用的模型全名
- 'base_url': 模型的API接入点
- 'api_key': (可选) 如果此模型的Key与通用Key不同，请在此处指定
"""

MODELS_TO_TEST = [
    {
        'name': "deepseek-v3",
        'api_model_name': "deepseek-chat",
        'base_url': "https://api.deepseek.com",
        'api_key': "sk-c42c790e216f4de3b2e890b730a0c946",
    },
    {
        'name': "qwen-max",
        'api_model_name': "qwen-max",
        'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'api_key': "sk-ca4ad778f16e453e8d21c489242ea180",
    },
    {
        'name': "ollama-qwen3-4B",
        'api_model_name': "qwen3:4b",
        'base_url': "http://localhost:11434/v1", 
        'api_key': "",
    },
    
]


# 2. 核心功能函数
def load_prompts(prompt_dir):
    """从指定的文本文件中加载所有提示词"""
    if not os.path.exists(prompt_dir):
        print(f"错误：提示词文件夹 '{prompt_dir}' 不存在。")
        return {}
    prompts = {}
    for filename in sorted(os.listdir(prompt_dir)):
        if filename.endswith('.txt'):
            prompt_id = re.search(r'(\d+)', filename)
            if not prompt_id:
                print(f"警告：文件 '{filename}' 不符合预期格式，跳过。")
                continue
            prompt_id = prompt_id.group(1)
            with open(os.path.join(prompt_dir, filename), 'r', encoding='utf-8') as f:
                prompts[prompt_id] = f.read()
            
    print(f"已成功从 '{prompt_dir}' 加载 {len(prompts)} 个提示词。")
    return prompts

def load_news_content_from_file(filepath):
    """从指定的文件中加载新闻内容"""
    if not os.path.exists(filepath):
        print(f"错误：新闻文件 '{filepath}' 不存在。")
        return None
    
    # 按分隔符分割标题和正文
    with open(filepath, 'r', encoding='utf-8') as f:
        parts = re.split(r'==================================================', f.read().strip())
        content = parts[1].strip() if len(parts) > 1 else None
    
    if not content:
        print(f"警告：文件 '{filepath}' 内容为空。")
        return None
    
    return content

def call_llm_api(client, model_name, system_prompt, user_prompt, retries=3, delay=5):
    """调用大模型API并返回结果(OPENAI 格式)"""
    for attempt in range(retries):
        try:
            print("    正在调用API...")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,  # 禁用流式输出
                temperature=0.4, # 设置较低的温度以保证输出的稳定性
                extra_body={"enable_thinking": False} if "qwen" in model_name.lower() else None
            )
            print("    API调用成功。")
            return completion.choices[0].message.content
        except Exception as e:
            print(f"    API调用失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"    将在 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("    所有重试均失败。")
                return f"ERROR: API call failed after {retries} retries. Last error: {e}"


def main():
    """主执行函数"""
    if not os.path.exists(INPUT_NEWS_DIR):
        print(f"错误：输入新闻文件夹 '{INPUT_NEWS_DIR}' 不存在。")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: {OUTPUT_DIR}")

    # 从外部文件加载提示词
    prompts = load_prompts(PROMPT_DIR)
    if not prompts:
        return

    news_files = sorted([f for f in os.listdir(INPUT_NEWS_DIR) if f.endswith('.txt')])

    for news_filename in news_files:
        # 从文件名提取新闻序号
        news_id_match = re.search(r'(\d+)\.txt', news_filename)
        if not news_id_match:
            continue
        news_id = news_id_match.group(1)
        
        news_filepath = os.path.join(INPUT_NEWS_DIR, news_filename)
        news_content = load_news_content_from_file(news_filepath)

        print(f"\n=========================================")
        print(f"正在处理新闻: {news_filename} (ID: {news_id})")
        print(f"=========================================")

        for model_config in MODELS_TO_TEST:
            model_name_for_file = model_config['name']
            api_model_name = model_config['api_model_name']
            base_url = model_config['base_url']
            
            # 优先使用模型特定key，否则使用通用key
            api_key = model_config.get('api_key', GENERAL_API_KEY)

            # 初始化API客户端
            client = OpenAI(api_key=api_key, base_url=base_url)

            print(f"\n  [模型: {model_name_for_file}]")

            for prompt_id in sorted(prompts.keys()):
                prompt_name = f"Prompt-{prompt_id}"
                print(f"\n    -> 正在使用提示词方案 {prompt_id} ({prompt_name})")
                
                # --- 构造输出文件名 ---
                output_filename = f"{STU_NAME}_事件要素_{model_name_for_file}_{news_id}_{prompt_id}.txt"
                output_filepath = os.path.join(OUTPUT_DIR ,output_filename)

                if os.path.exists(output_filepath):
                    print(f"      文件已存在，跳过: {output_filename}")
                    continue

                result_content = ""
                # --- 处理方案一至五 ---
                system_prompt = prompts[prompt_id]
                user_prompt = f"\n--- \n**开始分析:**\n现在，请分析以下新的“输入新闻文本”，并生成对应的Markdown输出。\n\n**输入新闻文本:**\n\n{news_content}"
                result_content = call_llm_api(client, api_model_name, system_prompt, user_prompt)

                # --- 保存结果 ---
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(result_content)
                    print(f"      结果已保存至: {output_filename}")
                except Exception as e:
                    print(f"      保存文件失败: {e}")
                
                # API调用之间增加延时
                time.sleep(3) 

    print("\n所有任务已完成！")


if __name__ == "__main__":
    main()