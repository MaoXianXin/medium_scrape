import os
from dialog_module.base import OneTimeDialogModule
from dialog_module.utils import read_template_from_file, read_article_from_file, create_custom_llm

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
        template_variables={"article_text": ""}  # 预先定义模板变量
    )
    
    # 生成文章总结
    summary = article_summarizer.process(
        article_text=article_text  # 直接传递article_text作为关键字参数
    )
    
    return summary

# 示例使用
if __name__ == "__main__":
    # 指定文件路径
    file_path = "/home/mao/workspace/medium_scrape/articles/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"
    
    # 读取文章内容
    article_text = read_article_from_file(file_path)
    
    # 生成文章总结
    summary = generate_article_summary(article_text)
    
    # 打印总结结果
    print("文章总结:")
    print("-" * 50)
    print(summary)