import requests
import json


DIFY_API_URL = "http://localhost/v1/chat-messages" 
DIFY_API_KEY = "app-f4wUTJc4ULK2yYUZPdMlVZVh" 


def load_questions_from_file(filename):
    questions = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Q:"):
                    # 提取 Q: 和 A: 之间的内容
                    q_part = line.split("A:")[0]
                    question = q_part.replace("Q:", "").strip()
                    if question:
                        questions.append(question)
    except IOError as e:
        print(f"读取文件时出错: {e}")
    return questions

def extract_thinking_answer(text):
    """分离 <think> 和 answer 部分"""
    if "<think>" in text and "</think>" in text:
        think_part = text.split("<think>")[1].split("</think>")[0].strip()
        answer_part = text.split("</think>")[-1].strip()
        return think_part, answer_part
    else:
        return "", text.strip()  # 如果没有 <think> 部分，直接返回答案部分

def query_dify_agent(question):
    """
    调用 Dify API 并返回模型的回答。
    """
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    # Dify API 请求体
    # 使用 "blocking" 模式来获取一次性完整回复
    # "user" ID 用于让 Dify 区分对话，这里用一个固定的ID即可
    payload = {
        "inputs": {},
        "query": question,
        "user": "dify-python-test666", 
        "response_mode": "blocking",
    }

    try:
        response = requests.post(DIFY_API_URL, headers=headers, json=payload, timeout=240)
        response.raise_for_status() 
        data = response.json()
        
        # 提取并返回答案
        answer = data.get("answer", "错误：未在回复中找到 'answer' 字段。")
        return extract_thinking_answer(answer)

    except requests.exceptions.RequestException as e:
        print(f"请求 Dify API 时发生网络错误: {e}")
        return f"API请求失败: {e}"
    except Exception as e:
        print(f"处理请求时发生未知错误: {e}")
        return f"未知错误: {e}"

def main(questions_file="private.txt", output_filename = "simple_basline_public.json"):
    """
    主函数，加载问题并调用 Dify API 处理。
    """
    questions = load_questions_from_file(questions_file)
    if not questions:
        print("未找到有效的问题，请检查输入文件。")
        return

    print(f"共找到 {len(questions)} 个问题。")
    
    qa_results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] 正在提问: {q}")
        
        thinking, answer = query_dify_agent(q)
        print(f"Dify 的回答: {answer}")

        # 构建问答对字典
        qa_pair = {
            "question": q,
            # "thinking": thinking, 
            "answer": answer 
        }
        # 将结果添加到列表中
        qa_results.append(qa_pair)

    # 将最终结果写入 JSON 文件
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=4)
        print(f"\n所有问题处理完毕！结果已成功保存到文件: {output_filename}")
    except IOError as e:
        print(f"写入文件时出错: {e}")
    

if __name__ == "__main__":
    main("public.txt", "strong_basline_public.json")