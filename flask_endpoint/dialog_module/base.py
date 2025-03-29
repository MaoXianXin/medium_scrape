from langchain.prompts import ChatPromptTemplate
from langchain.chat_models.base import BaseChatModel
from typing import Dict, Any, Optional, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# 设置日志
logger = logging.getLogger(__name__)

class OneTimeDialogModule:
    """
    通用一次性对话模块，接收文本内容、提示词模板和LLM Chat实例
    """
    
    def __init__(
        self, 
        llm: BaseChatModel,
        prompt_template: str,
        template_variables: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ):
        """
        初始化对话模块
        
        Args:
            llm: LLM Chat实例
            prompt_template: 提示词模板
            template_variables: 模板变量默认值（可选）
            max_retries: LLM调用失败时的最大重试次数
        
        Raises:
            ValueError: 如果提供的参数无效
        """
        if not isinstance(llm, BaseChatModel):
            raise ValueError("llm必须是BaseChatModel的实例")
        if not prompt_template or not isinstance(prompt_template, str):
            raise ValueError("prompt_template必须是非空字符串")
            
        self.llm = llm
        self.prompt_template = prompt_template
        self.template_variables = template_variables or {}
        self.max_retries = max_retries
        
        # 创建提示词模板
        try:
            self.chat_prompt = ChatPromptTemplate.from_template(prompt_template)
        except Exception as e:
            logger.error(f"创建提示词模板失败: {e}")
            raise ValueError(f"提示词模板格式无效: {e}")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt):
        """使用重试机制调用LLM"""
        return self.llm.invoke(prompt)
        
    def process(self, input_text: str = "", **kwargs) -> Union[str, None]:
        """
        处理输入文本并返回LLM响应
        
        Args:
            input_text: 用户输入的文本内容，默认为空字符串
            **kwargs: 其他模板变量
        
        Returns:
            LLM的响应文本，如果处理失败则返回None
        """
        try:
            # 合并默认变量和传入的变量
            variables = self.template_variables.copy()
            variables.update(kwargs)
            
            # 只有当input_text不为空时才添加到变量中
            if input_text:
                variables["input_text"] = input_text
            
            # 使用模板生成提示词
            prompt = self.chat_prompt.format_messages(**variables)
            
            # 调用LLM获取响应
            response = self._call_llm(prompt)
            
            return response.content
        except Exception as e:
            logger.error(f"处理请求时发生错误: {e}")
            return None