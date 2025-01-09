import os
from utils import OpenAIClient, KnowledgeFrameworkGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    
    def _process_single_file(self, input_dir: str, output_dir: str, filename: str) -> tuple[str, str]:
        """处理单个文件并返回结果"""
        article_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            return filename, "已存在"
            
        try:
            article_content = read_file(article_path)
            knowledge_framework = self.generate_single_framework(article_content)
            
            if not knowledge_framework or not knowledge_framework.strip():
                return filename, "框架为空"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(knowledge_framework)
            
            return filename, "成功"
            
        except Exception as e:
            return filename, f"错误: {str(e)}"

    def batch_generate_frameworks(self, input_dir: str, output_dir: str, max_workers: int = 4) -> None:
        """并发批量生成知识框架"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有需要处理的txt文件
        files_to_process = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self._process_single_file, input_dir, output_dir, filename): filename
                for filename in files_to_process
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                filename, status = future.result()
                if status == "成功":
                    print(f"已生成知识框架：{filename}")
                elif status == "已存在":
                    print(f"知识框架已存在，跳过生成：{filename}")
                elif status == "框架为空":
                    print(f"警告：{filename} 生成的知识框架为空，跳过保存")
                else:
                    print(f"处理文件 {filename} 时发生错误: {status}")

class ServiceConfig:
    """服务配置类"""
    CONFIGS = {
        "knowledge_framework": {
            "prompt_file_path": "Prompt知识框架.txt",
            "system_prompt": "你是一位专业的知识框架生成专家",
            "input_dir": "articles",
            "output_dir": "knowledge_frameworks"
        },
        "article_summary": {
            "prompt_file_path": "Prompt文章总结.txt",
            "system_prompt": "你是一位专业的文章分析专家",
            "input_dir": "articles",
            "output_dir": "article_summaries"
        }
    }

    @classmethod
    def get_config(cls, service_type: str) -> dict:
        """获取指定服务类型的配置"""
        if service_type not in cls.CONFIGS:
            raise ValueError(f"不支持的服务类型: {service_type}")
        return cls.CONFIGS[service_type]

def main():
    # API配置
    api_config = {
        "api_key": "sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        "base_url": "https://www.gptapi.us/v1",
        "model_name": "gpt-4o-mini"
    }
    
    # 选择服务类型
    service_type = "article_summary"  # 或 "knowledge_framework"
    config = ServiceConfig.get_config(service_type)
    
    service = KnowledgeFrameworkService(
        **api_config,
        prompt_file_path=config["prompt_file_path"],
        system_prompt=config["system_prompt"]
    )
    
    service.batch_generate_frameworks(
        input_dir=config["input_dir"],
        output_dir=config["output_dir"],
        max_workers=8
    )

if __name__ == "__main__":
    main()