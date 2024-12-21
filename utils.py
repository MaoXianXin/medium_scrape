from openai import OpenAI
from typing import List, Dict, Optional

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