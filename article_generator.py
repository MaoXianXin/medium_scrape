from pathlib import Path
from typing import Optional, List, Tuple
from utils import OpenAIClient
from concurrent.futures import ThreadPoolExecutor

class ArticleGenerator:
    def __init__(
        self, 
        openai_client: OpenAIClient,
        summaries_dir: str = "summaries", 
        articles_dir: str = "articles",
        output_dir: str = "generated_articles"
    ):
        """
        初始化文章生成器
        
        Args:
            openai_client: OpenAI客户端实例
            summaries_dir: 知识框架目录路径
            articles_dir: 原始文章目录路径
            output_dir: 生成文章的输出目录路径
        """
        self.client = openai_client
        self.summaries_dir = Path(summaries_dir)
        self.articles_dir = Path(articles_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
            
    def _generate_prompt(self, framework_file: str, article_file: str) -> str:
        """生成完整的提示词"""
        framework_path = self.summaries_dir / framework_file
        article_path = self.articles_dir / article_file
        
        framework_content = self.read_file(framework_path)
        article_content = self.read_file(article_path)
        
        return f"""知识框架:
---
{framework_content}
---

原始文章:
---
{article_content}
---

根据给定的知识框架和原始文章，在知识框架图作为指导的前提下，进行文章创作"""

    def save_article(self, content: str, filename: str) -> Path:
        """
        保存生成的文章
        
        Args:
            content: 文章内容
            filename: 输出文件名
        Returns:
            Path: 保存文件的路径
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path

    def generate(self, framework_file: str, article_file: str, output_file: Optional[str] = None) -> str:
        """
        根据知识框架和原始文章生成新文章并保存
        
        Args:
            framework_file: 知识框架文件名
            article_file: 原始文章文件名
            output_file: 输出文件名，如果为None则使用默认名称
        Returns:
            str: 生成的文章内容
        """
        content = self._generate_prompt(framework_file, article_file)
        messages = [{"role": "user", "content": content}]
        
        system_prompt = """你是一个专业的文章创作者。请基于给定的知识框架作为指导，
        参考原始文章的内容，创作一篇结构清晰、逻辑严密的新文章。确保新文章符合知识框架的整体结构，
        同时保持内容的原创性和可读性。"""
        
        response = self.client.get_completion(messages, system_prompt)
        
        if output_file is None:
            output_file = Path(article_file).name
        
        self.save_article(response.content, output_file)
        return response.content

    def generate_batch(self, max_workers: int = 3) -> List[Tuple[str, str]]:
        """
        并行批量处理summaries和articles目录下的所有配对文件
        
        Args:
            max_workers: 最大并行工作线程数
            
        Returns:
            List[Tuple[str, str]]: 包含(文件名, 生成内容)的列表
        """
        results = []
        framework_files = list(self.summaries_dir.glob("*.txt"))
        total_files = len(framework_files)
        
        print(f"开始并行批量处理，共发现 {total_files} 个文件，使用 {max_workers} 个工作线程")
        
        def process_file(framework_path: Path) -> Tuple[str, str]:
            framework_name = framework_path.name
            article_path = self.articles_dir / framework_name
            output_path = self.output_dir / framework_name
            
            if not article_path.exists():
                print(f"警告: 未找到对应的文章文件 {framework_name}")
                return framework_name, ""
                
            if output_path.exists():
                print(f"跳过已存在的文章: {framework_name}")
                return framework_name, self.read_file(output_path)
                
            print(f"正在生成新文章: {framework_name}")
            content = self.generate(framework_name, framework_name)
            print(f"文章生成完成: {framework_name}")
            return framework_name, content
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, path) for path in framework_files]
            for future in futures:
                filename, content = future.result()
                if content:  # 只添加成功生成的结果
                    results.append((filename, content))
                
        return results

if __name__ == "__main__":
    # 初始化OpenAI客户端
    client = OpenAIClient(
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        model_name="gpt-4o-mini"
    )

    # 初始化文章生成器并设置8个并行线程
    article_generator = ArticleGenerator(
        client,
        summaries_dir="knowledge_frameworks",
        articles_dir="articles",
        output_dir="generated_articles"
    )
    results = article_generator.generate_batch(max_workers=8)
    print(f"已处理 {len(results)} 篇文章")