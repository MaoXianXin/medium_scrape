import os
from utils import OpenAIClient, KnowledgeFrameworkGenerator

"""
knowledge_framework_generator.py 是知识框架生成器的实现。
它使用OpenAI API来生成知识框架，并保存到指定的目录中。
Prompt知识框架.txt
system_prompt
"""

def read_file(file_path: str) -> str:
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

class KnowledgeFrameworkService:
    """知识框架生成服务类"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str, 
                 prompt_file_path: str = "Prompt知识框架.txt",
                 system_prompt: str = "你是一位专业的知识框架生成专家"):
        self.ai_client = OpenAIClient(
            api_key=api_key,
            base_url=base_url, 
            model_name=model_name
        )
        self.framework_generator = KnowledgeFrameworkGenerator(self.ai_client)
        self.prompt_file_path = prompt_file_path
        self.system_prompt = system_prompt
        self.framework_prompt = self._read_framework_prompt()
    
    def _read_framework_prompt(self) -> str:
        """读取知识框架提示词"""
        return read_file(self.prompt_file_path)
    
    def generate_single_framework(self, article_content: str) -> str:
        """为单篇文章生成知识框架"""
        return self.framework_generator.generate(
            article_content,
            self.framework_prompt,
            system_prompt=self.system_prompt
        )
    
    def batch_generate_frameworks(self, input_dir: str, output_dir: str) -> None:
        """批量生成知识框架"""
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if not filename.endswith(".txt"):
                continue
                
            article_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if os.path.exists(output_path):
                print(f"知识框架已存在，跳过生成：{output_path}")
                continue
            
            try:
                article_content = read_file(article_path)
                knowledge_framework = self.generate_single_framework(article_content)
                
                if not knowledge_framework or not knowledge_framework.strip():
                    print(f"警告：{filename} 生成的知识框架为空，跳过保存")
                    continue
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(knowledge_framework)
                
                print(f"已生成知识框架：{output_path}")
                
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {str(e)}")

"""
# 知识框架生成器
prompt_file_path="Prompt知识框架.txt"
system_prompt="你是一位专业的知识框架生成专家"

# 文章总结
prompt_file_path="Prompt文章总结.txt"
system_prompt="你是一位专业的文章分析专家"
"""

def main():
    service = KnowledgeFrameworkService(
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        model_name="gpt-4o-mini",
        prompt_file_path="Prompt知识框架.txt",
        system_prompt="你是一位专业的知识框架生成专家"
    )
    service.batch_generate_frameworks("articles", "summaries")

if __name__ == "__main__":
    main()