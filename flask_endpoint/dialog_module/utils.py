from langchain_openai import ChatOpenAI


def read_template_from_file(template_path):
    """
    从文件中读取提示词模板
    
    Args:
        template_path: 模板文件路径
        
    Returns:
        模板内容字符串
    """
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取模板文件时出错: {e}")
        return "无法读取模板文件内容"

def read_article_from_file(file_path):
    """
    从文件中读取文章内容
    
    Args:
        file_path: 文章文件路径
        
    Returns:
        文章内容字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return "无法读取文件内容"

def create_custom_llm(
    model_name: str = "deepseek-r1",
    temperature: float = 0.1,
    base_url: str = "https://www.gptapi.us/v1",
    api_key: str = "sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7"
):
    """
    创建自定义LLM实例
    
    Args:
        model_name: 模型名称
        temperature: 温度参数
        base_url: API基础URL
        api_key: API密钥
        
    Returns:
        ChatOpenAI实例
    """
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key
    )
