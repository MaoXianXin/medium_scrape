from langchain_openai import ChatOpenAI
from dialog_module.base import OneTimeDialogModule
import os

# 使用自定义模型服务
custom_llm = ChatOpenAI(
    model_name="deepseek-r1",
    temperature=0.1,
    base_url="https://www.gptapi.us/v1",
    api_key="sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7"
)

# 从外部文件读取提示词模板
def read_template_from_file(template_path):
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取模板文件时出错: {e}")
        return "无法读取模板文件内容"

# 指定模板文件路径
template_path = os.path.join(os.path.dirname(__file__), "templates", "article_summary_template.txt")
summary_template = read_template_from_file(template_path)

# 创建文章总结模块实例
article_summarizer = OneTimeDialogModule(
    llm=custom_llm,
    prompt_template=summary_template,
    template_variables={"article_text": ""}  # 应该预先定义模板变量
)

# 从文件读取文章内容
def read_article_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return "无法读取文件内容"

# 指定文件路径
file_path = "/home/mao/workspace/medium_scrape/articles/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"
article_text = read_article_from_file(file_path)

# 生成文章总结
summary = article_summarizer.process(
    article_text=article_text  # 直接传递article_text作为关键字参数
)

# 打印总结结果
print("文章总结:")
print("-" * 50)
print(summary)