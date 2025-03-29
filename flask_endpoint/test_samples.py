from langchain_openai import ChatOpenAI
from dialog_module.base import OneTimeDialogModule


# 使用自定义模型服务
custom_llm = ChatOpenAI(
    model_name="deepseek-r1",
    temperature=0.1,
    base_url="https://www.gptapi.us/v1",
    api_key="sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7"
)

# 创建提示词模板
template = """
你是一个专业的文本分析助手。请分析以下文本内容：

{input_text}

请从以下角度进行分析：
1. {aspect_1}
2. {aspect_2}
3. {aspect_3}

分析深度：{depth}
"""

# 创建对话模块实例
dialog_module = OneTimeDialogModule(
    llm=custom_llm,
    prompt_template=template,
    template_variables={
        "aspect_1": "主题和核心观点",
        "aspect_2": "论证结构和逻辑",
        "aspect_3": "语言风格和表达方式",
        "depth": "详细"
    }
)

# 从文件读取输入文本
def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return "无法读取文件内容"

# 指定文件路径
file_path = "/home/mao/workspace/medium_scrape/articles/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"  # 可以根据需要修改文件路径
input_text = read_text_from_file(file_path)

response = dialog_module.process(
    input_text,
    # 可以覆盖默认变量
    aspect_3="情感倾向和态度",
    depth="简要"
)

print(response)