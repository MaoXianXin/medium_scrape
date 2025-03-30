import os
import re
from dialog_module.base import OneTimeDialogModule
from dialog_module.utils import read_template_from_file, read_article_from_file, create_custom_llm
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict

# 定义标签结构模型到文件顶部
class TagsModel(BaseModel):
    技术标签: List[str] = Field(description="文章中提到的具体技术、算法或方法名称")
    主题标签: List[str] = Field(description="反映文章的整体主题和研究领域")
    应用标签: List[str] = Field(description="技术的应用场景或解决的问题")
    统一标签: List[str] = Field(description="所有标签的合集，前面加#号")
    标签解释: Dict[str, str] = Field(description="每个标签的选择理由简述")

# 使用自定义模型服务
custom_llm = create_custom_llm()

def generate_article_summary(article_text, template_path=None):
    """
    生成文章总结的函数
    
    参数:
        article_text: 文章内容文本
        template_path: 提示词模板路径，如果为None则使用默认路径
        
    返回:
        生成的文章总结
    """
    # 如果未提供模板路径，使用默认路径
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__), "templates", "article_summary_template.txt")
    
    # 读取提示词模板
    summary_template = read_template_from_file(template_path)
    
    # 创建文章总结模块实例
    article_summarizer = OneTimeDialogModule(
        llm=custom_llm,
        prompt_template=summary_template,
        template_variables={}  # 不预设任何模板变量
    )
    
    # 生成文章总结
    summary = article_summarizer.process(
        article_text=article_text  # 通过kwargs传递
    )
    
    return summary

def extract_tags_from_summary(summary, template_path=None):
    """
    从文章总结中提取关键词作为标签
    
    参数:
        summary: 文章总结文本
        template_path: 提示词模板路径，如果为None则使用默认路径
        
    返回:
        提取的标签JSON对象，包含技术标签、主题标签、应用标签、统一标签和标签解释
    """
    # 如果未提供模板路径，使用默认路径
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__), "templates", "extract_tags_template.txt")
    
    # 读取提示词模板
    tags_template = read_template_from_file(template_path)
    
    # 创建JSON解析器
    parser = JsonOutputParser(pydantic_object=TagsModel)
    
    # 创建标签提取模块实例
    tag_extractor = OneTimeDialogModule(
        llm=custom_llm,
        prompt_template=tags_template,
        template_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 提取标签
    tags_text = tag_extractor.process(
        summary=summary  # 通过kwargs传递
    )
    
    # 解析JSON结果
    try:
        tags = parser.parse(tags_text)
        return tags
    except Exception as e:
        print(f"JSON解析错误: {e}")
        return tags_text  # 如果解析失败，返回原始文本

# 示例使用
if __name__ == "__main__":
    # 指定文件路径
    file_path = "/home/mao/workspace/medium_scrape/articles/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"
    
    # 读取文章内容
    article_text = read_article_from_file(file_path)
    
    # 生成文章总结
    summary = generate_article_summary(article_text)
    # 过滤掉<think>...</think>内容
    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
    
    # 打印总结结果
    print("文章总结:")
    print("-" * 50)
    print(summary)
    
    # 从总结中提取标签
    tags = extract_tags_from_summary(summary)
    
    # 打印标签
    print("\n文章标签:")
    print("-" * 50)
    print(tags)
    
    # 可以直接访问结构化数据
    if isinstance(tags, dict) or hasattr(tags, "__dict__"):
        print("\n技术标签:", tags.技术标签 if hasattr(tags, "技术标签") else tags.get("技术标签", []))
        print("主题标签:", tags.主题标签 if hasattr(tags, "主题标签") else tags.get("主题标签", []))