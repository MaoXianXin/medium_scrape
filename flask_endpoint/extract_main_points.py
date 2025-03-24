from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_community.callbacks.manager import get_openai_callback

def extract_article_main_points(article_summary, api_key=None, return_tokens=False, temperature=0.1, base_url="https://zzzzapi.com/v1", model="gpt-4o-mini"):
    """
    Extract main points from an article summary using a specialized prompt.
    
    Args:
        article_summary (str): The summary of the article to extract main points from
        api_key (str, optional): OpenAI API key
        return_tokens (bool, optional): Whether to return token usage information
        temperature (float, optional): Temperature for the model
        base_url (str, optional): Base URL for the API
        model (str, optional): Model to use
        
    Returns:
        str or tuple: JSON string with main points or tuple of (JSON string, token_info)
    """
    # Create the extraction prompt template
    extraction_template = """
    你是一个精确的文本分析工具。请从以下文章总结中提取"核心观点提炼"部分的所有主要观点，并将它们格式化为JSON数组输出。

    具体要求：
    1. 只提取"2. 核心观点提炼"部分中以"主要观点"开头的内容
    2. 对于每个主要观点，提取冒号后面的完整内容
    3. 不要包含编号（如"主要观点1:"）在结果中，只提取观点内容本身
    4. 以JSON数组格式返回所有主要观点，键名为"main_points"
    5. 不要添加任何额外的解释或评论，只返回JSON格式的结果
    6. 不要将JSON结果包含在任何代码块中，直接输出原始JSON文本

    示例输入:

    1. 文章概述:
       - 主题: 人工智能在商业中的应用
       - 作者的写作目的: 探讨AI如何改变企业运营

    2. 核心观点提炼:
       - 主要观点1: 人工智能技术的快速发展正在重塑多个行业的业务模式和工作流程。
       - 主要观点2: 企业需要制定系统性策略来整合AI技术，而非仅将其视为独立工具。
       - 主要观点3: AI实施的成功依赖于跨部门合作和持续的员工培训。

    示例输出:

    {{
      "main_points": [
        "人工智能技术的快速发展正在重塑多个行业的业务模式和工作流程。",
        "企业需要制定系统性策略来整合AI技术，而非仅将其视为独立工具。",
        "AI实施的成功依赖于跨部门合作和持续的员工培训。"
      ]
    }}

    现在，请从以下文章总结中提取所有主要观点并以指定格式返回:

    {article_summary}
    """
    
    # Create prompt template
    prompt = PromptTemplate(
        template=extraction_template,
        input_variables=["article_summary"]
    )
    
    # Create language model
    llm = ChatOpenAI(
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Build the chain
    chain = prompt | llm | output_parser
    
    # Use callback to track token usage
    with get_openai_callback() as cb:
        # Execute the chain
        result = chain.invoke({"article_summary": article_summary})
        
        # Filter out <think>...</think> content
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        
        # If token information is requested
        if return_tokens:
            token_info = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost
            }
            return result, token_info
    
    return result

# Example usage
if __name__ == "__main__":
    # Sample article summary
    sample_summary = """
    1. 文章概述:
       - 主题: 人工智能在商业中的应用
       - 作者的写作目的: 探讨AI如何改变企业运营

    2. 核心观点提炼:
       - 主要观点1: 人工智能技术的快速发展正在重塑多个行业的业务模式和工作流程。
       - 主要观点2: 企业需要制定系统性策略来整合AI技术，而非仅将其视为独立工具。
       - 主要观点3: AI实施的成功依赖于跨部门合作和持续的员工培训。
    """
    
    # Extract main points with token tracking
    main_points, token_info = extract_article_main_points(
        article_summary=sample_summary,
        api_key="sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
        return_tokens=True,
        temperature=0.1,
        base_url="https://www.gptapi.us/v1",
        model="deepseek-r1"
    )
    print("\n--- Main Points ---")
    print(main_points)
    print("\n--- Token Usage ---")
    print(f"Total Tokens: {token_info['total_tokens']}")
    print(f"Prompt Tokens: {token_info['prompt_tokens']}")
    print(f"Completion Tokens: {token_info['completion_tokens']}")
    print(f"Total Cost: ${token_info['total_cost']:.6f}") 