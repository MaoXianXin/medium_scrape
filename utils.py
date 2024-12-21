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