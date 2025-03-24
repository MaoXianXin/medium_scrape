from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_community.callbacks.manager import get_openai_callback

def summarize_article(article_text, api_key=None, return_tokens=False, temperature=0.1, base_url="https://zzzzapi.com/v1", model="gpt-4o-mini"):
    # 创建提示模板
    summary_template = """
    1. 文章概述:
       - 主题:
       - 作者的写作目的:
       - 文章的整体结构:
    
    2. 核心观点提炼:
       - 主要观点1: [这里填写第一个主要观点，用完整陈述句表达，应当包含作者的核心主张或论点]
       - 主要观点2: [这里填写第二个主要观点，确保与第一个观点不重复，代表文章的另一个关键论点]
       - 主要观点3: [如果文章包含第三个主要观点，请在此处填写]
       - [如果有更多主要观点，请按照相同格式继续添加]
    
    3. 论据分析:
       对于每个核心观点,请提供:
       - 支持论据:
       - 论据类型(如数据、案例、专家观点等):
       - 论据的有效性评估:
    
    4. 副观点和支持性细节:
       - 副观点1:
       - 副观点2:
       - 重要细节:
    
    5. 关键词和重复概念:
       列出文章中反复出现或强调的关键词和概念。
    
    核心观点提炼示例:
    示例1:
    2. 核心观点提炼:
       - 主要观点1: 人工智能技术的快速发展正在重塑多个行业的业务模式和工作流程。
       - 主要观点2: 企业需要制定系统性策略来整合AI技术，而非仅将其视为独立工具。
       - 主要观点3: AI实施的成功依赖于跨部门合作和持续的员工培训。
    
    示例2:
    2. 核心观点提炼:
       - 主要观点1: 可持续发展不仅是环境责任，也是企业长期生存和竞争力的关键要素。
       - 主要观点2: 消费者越来越倾向于支持具有强烈环保承诺的品牌和产品。
       - 主要观点3: 建立真正可持续的商业模式需要从供应链到产品生命周期的全面创新。
       - 主要观点4: 政府政策和行业标准正在加速企业向可持续实践转型。
    
    根据上述的方法和示例，对以下文章进行全面的内容提炼和分析，请用中文回复，确保核心观点提炼部分的格式与示例一致。直接输出markdown格式的内容，不要将内容包含在markdown代码块中，也不要在各个章节之间添加分隔线:
    
    {article_text}
    """
    
    # 创建提示模板
    prompt = PromptTemplate(
        template=summary_template,
        input_variables=["article_text"]
    )
    
    # 创建语言模型
    model = ChatOpenAI(
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    
    # 创建输出解析器
    output_parser = StrOutputParser()
    
    # 构建处理链
    chain = prompt | model | output_parser
    
    # 使用回调来跟踪token使用情况
    with get_openai_callback() as cb:
        # 执行链并返回结果
        summary = chain.invoke({"article_text": article_text})
        
        # 过滤掉<think>...</think>内容
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        
        # 如果需要返回token信息
        if return_tokens:
            token_info = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost
            }
            return summary, token_info
    
    return summary

# 使用示例
if __name__ == "__main__":
    # 示例文章
    sample_article = """
    文章内容
    """
    
    # 获取总结和token信息
    summary, token_info = summarize_article(
        article_text=sample_article,
        api_key="sk-UxCneocSvk83jPkSmDRyYZA2zLWiAX1Ds71JVK72IqH1DiR6",
        return_tokens=True,
        temperature=0.2,
        base_url="https://zzzzapi.com/v1",
        model="deepseek-r1"
    )
    print(summary)
    print("\n--- Token 使用情况 ---")
    print(f"总Token数: {token_info['total_tokens']}")
    print(f"提示Token数: {token_info['prompt_tokens']}")
    print(f"完成Token数: {token_info['completion_tokens']}")
    print(f"总成本: ${token_info['total_cost']:.6f}")