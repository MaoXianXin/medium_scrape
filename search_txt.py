from openai import OpenAI
import os
from tqdm import tqdm

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_article_prompt(article_path, search_topic):
    article = read_file(article_path)
    prompt = (
        f"请仔细阅读以下文章，判断文章是否讨论了{search_topic}。"
        "\n\n请只回答'是'或'否'。"
        "\n\n文章内容：\n" + article
    )
    return prompt

def update_knowledge_framework(original_content, new_content, search_topic, use_full_text=False):
    content_type = "完整文章" if use_full_text else "关键信息"
    prompt = (
        "请基于以下内容更新知识框架：\n\n"
        "当前知识框架：\n"
        f"{original_content}\n\n"
        f"新发现的相关{content_type}：\n"
        f"{new_content}\n\n"
        f"请分析新内容中与{search_topic}相关的内容，"
        "在适当的位置补充或修改原有知识框架。"
        "保持原有框架结构，仅添加或更新相关内容。"
        "如果发现新的重要分类，可以适当添加新的章节。"
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个专业的知识框架构建专家，善于整合和组织知识。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content.strip()

client = OpenAI(
    api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
    base_url="https://www.gptapi.us/v1"
)

def analyze_article(article_path, search_topic):
    prompt = create_article_prompt(article_path, search_topic)
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个专业的文章分析专家。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content.strip()

def extract_relevant_content(article_path, search_topic):
    prompt = (
        f"请仔细阅读以下文章，提取与'{search_topic}'相关的关键信息和段落。"
        "\n请以摘要的形式组织信息，突出重点内容。"
        "\n\n文章内容：\n" + read_file(article_path)
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个专业的文章分析专家，善于提取文章中的关键信息。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content.strip()

def main():
    search_topic = "大语言模型的推理框架"
    articles_dir = "summaries"
    original_articles_dir = "articles"
    knowledge_framework_path = "knowledge_framework.txt"
    
    # 用户选择更新方式
    update_framework = input("是否需要更新知识框架？(是/否): ").strip().lower() == "是"
    if update_framework:
        update_mode = input("请选择更新模式（1：使用提取的关键信息 2：使用完整文章内容）: ").strip()
        use_full_text = (update_mode == "2")
    
    # 读取当前知识框架
    current_framework = read_file(knowledge_framework_path)
    
    articles = [f for f in os.listdir(articles_dir) if f.endswith('.txt')]
    progress_bar = tqdm(articles, desc="Processing articles")
    result_files = []
    relevant_content_dict = {}  # 使用字典存储文件名和对应的相关内容
    
    # 分析文章并收集相关内容
    for article_file in progress_bar:
        article_path = os.path.join(articles_dir, article_file)
        try:
            answer = analyze_article(article_path, search_topic)
            if answer == "是":
                result_files.append(article_file)
                original_article_path = os.path.join(original_articles_dir, article_file)
                if os.path.exists(original_article_path):
                    content = extract_relevant_content(original_article_path, search_topic)
                    relevant_content_dict[article_file] = content
        except Exception as e:
            print(f"处理 {article_file} 时出错: {str(e)}")
    
    # 更新知识框架
    if relevant_content_dict and update_framework:
        print("\n开始更新知识框架...")
        current = current_framework
        
        for i, article_file in enumerate(result_files, 1):
            print(f"\n处理第 {i}/{len(result_files)} 篇文章的内容...")
            try:
                original_article_path = os.path.join(original_articles_dir, article_file)
                if os.path.exists(original_article_path):
                    if use_full_text:
                        # 使用完整文章内容
                        content = read_file(original_article_path)
                    else:
                        # 使用提取的关键信息
                        content = relevant_content_dict[article_file]
                    
                    current = update_knowledge_framework(
                        current, 
                        content, 
                        search_topic,
                        use_full_text
                    )
            except Exception as e:
                print(f"更新第 {i} 篇文章内容时出错: {str(e)}")
                continue
                
        write_file(knowledge_framework_path, current)
        print(f"\n知识框架已更新，保存至 {knowledge_framework_path}")
    elif relevant_content_dict and not update_framework:
        print("\n发现新的相关内容，但根据设置未更新知识框架")
    
    # 输出分析结果
    print(f"\n以下文件中提及了{search_topic}：")
    for article_file in result_files:
        print(f"\n文件名: {article_file}")
        try:
            original_article_path = os.path.join(original_articles_dir, article_file)
            if os.path.exists(original_article_path):
                print("\n相关内容摘要:")
                # 直接使用之前存储的内容，避免重复调用
                print(relevant_content_dict[article_file])
            else:
                print(f"未找到原始文章文件: {original_article_path}")
        except Exception as e:
            print(f"提取内容时出错: {str(e)}")

if __name__ == "__main__":
    main()
