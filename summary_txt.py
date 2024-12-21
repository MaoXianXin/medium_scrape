from openai import OpenAI
import os
from tqdm import tqdm

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_article_prompt(template_path, article_path):
    # 读取提示词模板和文章内容
    template = read_file(template_path)
    article = read_file(article_path)
    
    # 将文章内容添加到模板末尾
    return template + "\n\n" + article

# Allow custom API key and base URL
client = OpenAI(
    api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
    base_url="https://www.gptapi.us/v1"
)

def analyze_article(template_path, article_path, max_retries=3):
    # 创建完整的提示词
    prompt = create_article_prompt(template_path, article_path)
    
    for attempt in range(max_retries):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的文章分析专家，善于按照模板提取和分析文章的关键信息。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = completion.choices[0].message.content
        
        # 验证输出格式
        if validate_output_format(result):
            return result
        
        # 如果是最后一次尝试，则抛出异常
        if attempt == max_retries - 1:
            raise ValueError(f"在{max_retries}次尝试后，输出格式仍不符合要求")
        
        # 添加更明确的格式要求进行重试
        prompt = (
            "请严格按照以下格式输出：\n"
            "1. 必须包含\"第一部分：知识框架图\"和\"第二部分：核心概念定义\"两个部分\n"
            "2. 知识框架图必须使用树形结构（使用├或└符号）\n"
            "3. 核心概念定义必须使用\"概念: 定义\"的格式\n\n"
            + prompt
        )

def validate_output_format(text):
    # 检查基本结构
    if "第一部分：知识框架图" not in text or "第二部分：核心概念定义" not in text:
        return False
    
    # 分割两个部分
    parts = text.split("第二部分：核心概念定义")
    if len(parts) != 2:
        return False
    
    framework_part = parts[0].strip()
    definitions_part = parts[1].strip()
    
    # 检查知识框架图部分是否包含树形结构（至少有一个├或└符号）
    if not any(symbol in framework_part for symbol in ['├', '└']):
        return False
    
    # 检查核心概念定义部分的格式（每行应该是"概念: 定义"的格式）
    definitions = [line.strip() for line in definitions_part.split('\n') if line.strip()]
    for definition in definitions:
        if ': ' not in definition and ':' not in definition:
            return False
    
    return True

def main():
    # 设置文件路径
    template_path = "/home/mao/workspace/gpt_coding/提示词模版/信息整理/Prompt知识框架.txt"
    articles_dir = "articles"
    
    # 设置输出目录
    output_dir = "summaries"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有需要处理的文章
    articles = [f for f in os.listdir(articles_dir) if f.endswith('.txt')]
    
    # 使用tqdm创建进度条
    progress_bar = tqdm(articles)
    for article_file in progress_bar:
        article_path = os.path.join(articles_dir, article_file)
        output_name = f"{os.path.splitext(article_file)[0]}.txt"
        output_path = os.path.join(output_dir, output_name)
        
        # 使用tqdm.write来打印日志
        if os.path.exists(output_path):
            tqdm.write(f"跳过: {article_file}")
            continue
                
        tqdm.write(f"处理: {article_file}")
        try:
            result = analyze_article(template_path, article_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            tqdm.write(f"已完成: {article_file}")
        except Exception as e:
            tqdm.write(f"错误 {article_file}: {str(e)}")

if __name__ == "__main__":
    main()