import os
from utils import OpenAIClient, KnowledgeFrameworkGenerator

def read_file(file_path: str) -> str:
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # 初始化OpenAI客户端
    ai_client = OpenAIClient(
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        model_name="gpt-4o-mini"
    )
    
    # 初始化知识框架生成器
    framework_generator = KnowledgeFrameworkGenerator(ai_client)
    
    # 读取知识框架提示词
    framework_prompt = read_file("Prompt知识框架.txt")
    
    # 读取articles目录下的所有文章
    articles_dir = "articles"
    # 确保summaries目录存在
    os.makedirs("summaries", exist_ok=True)
    
    for filename in os.listdir(articles_dir):
        if filename.endswith(".txt"):
            article_path = os.path.join(articles_dir, filename)
            article_content = read_file(article_path)
            
            # 生成知识框架
            knowledge_framework = framework_generator.generate(
                article_content,
                framework_prompt,
                system_prompt="你是一个专业的知识框架生成专家"
            )
            
            # 保存结果到summaries目录，使用原始文件名
            output_path = os.path.join("summaries", filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(knowledge_framework)
            
            print(f"已生成知识框架：{output_path}")

if __name__ == "__main__":
    main()