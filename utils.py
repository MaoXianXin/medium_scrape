from openai import OpenAI
from typing import List, Dict, Optional, Union
import os

"""
utils.py 包含OpenAI客户端、知识框架生成器和文章筛选器的实现。
提示词共同工作，指导LLM模型完成知识框架生成和相关度计算等任务:
OpenAIClient类的get_completion函数中的system_prompt
KnowledgeFrameworkGenerator类的generate函数中的system_prompt跟messages
ArticleFilter类的calculate_relevance函数中的system_prompt跟messages
"""

class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str
    ):
        """
        初始化OpenAI客户端
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model_name: 模型名称
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        
    def get_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        获取OpenAI API响应
        
        Args:
            messages: 消息列表
            system_prompt: 可选的系统提示
        """
        messages_copy = messages.copy()
        
        if system_prompt:
            messages_copy.insert(0, {"role": "system", "content": system_prompt})
            
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_copy
        )
        
        return completion.choices[0].message

class KnowledgeFrameworkGenerator:
    def __init__(self, ai_client: OpenAIClient):
        """
        初始化知识框架生成器
        
        Args:
            ai_client: OpenAI客户端实例
        """
        self.ai_client = ai_client
        
    def generate(
        self, 
        article_content: str, 
        framework_prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        生成知识框架
        
        Args:
            article_content: 文章内容
            framework_prompt: 知识框架提示词模板
            system_prompt: 可选的系统提示词
            
        Returns:
            str: 生成的知识框架
        """
        messages = [
            {
                "role": "user",
                "content": f"请根据以下提示词和文章内容，生成规范的知识框架：\n\n"
                          f"提示词：\n{framework_prompt}\n\n"
                          f"文章内容：\n{article_content}"
            }
        ]
        
        response = self.ai_client.get_completion(
            messages=messages,
            system_prompt=system_prompt
        )
        return response.content

class ArticleFilter:
    def __init__(self, ai_client: OpenAIClient):
        """
        初始化文章筛选器
        
        Args:
            ai_client: OpenAI客户端实例
        """
        self.ai_client = ai_client
        
    def calculate_relevance(
        self,
        knowledge_framework: str,
        search_topic: str,
        system_prompt: Optional[str] = None
    ) -> float:
        """
        计算知识框架图与搜索主题的相关度分数
        
        Args:
            knowledge_framework: 知识框架内容
            search_topic: 搜索主题
            system_prompt: 可选的系统提示词
            
        Returns:
            float: 相关度分数(0-1)
        """
        # 提取第一部分：知识框架图
        framework_part = knowledge_framework.split("第二部分：核心概念定义")[0].strip()

        messages = [
            {
                "role": "user",
                "content": f"请分析以下知识框架图与搜索主题的相关度，返回一个0到1之间的分数。分数越高表示相关度越高。\n\n"
                          f"知识框架图：\n{framework_part}\n\n"
                          f"搜索主题：\n{search_topic}\n\n"
                          f"请只返回分数，例如：0.75"
            }
        ]
        
        response = self.ai_client.get_completion(
            messages=messages,
            system_prompt=system_prompt
        )
        return float(response.content)

    def filter_articles(
        self,
        summaries_dir: str,
        search_topic: str,
        threshold: float = 0.5,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        根据搜索主题筛选相关文章
        
        Args:
            summaries_dir: 知识框架文件目录
            search_topic: 搜索主题
            threshold: 相关度阈值
            system_prompt: 可选的系统提示词
            
        Returns:
            List[Dict[str, Union[str, float]]]: 筛选后的文章列表，每个字典包含：
                - filename (str): 文件名
                - relevance_score (float): 相关度分数
                - framework (str): 知识框架内容
        """
        filtered_articles = []
        
        # 遍历summaries目录下的所有txt文件
        for filename in os.listdir(summaries_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(summaries_dir, filename)
                
                # 读取知识框架内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    knowledge_framework = f.read()
                
                # 计算相关度分数
                relevance_score = self.calculate_relevance(
                    knowledge_framework=knowledge_framework,
                    search_topic=search_topic,
                    system_prompt=system_prompt
                )
                
                # 如果分数超过阈值，添加到结果列表
                if relevance_score >= threshold:
                    filtered_articles.append({
                        'filename': filename,
                        'relevance_score': relevance_score,
                        'framework': knowledge_framework
                    })
        
        # 按相关度分数降序排序
        filtered_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        return filtered_articles