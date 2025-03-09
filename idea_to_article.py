"""
flowchart TD
    A[开始] --> B[读取core_idea.txt文件中的核心观点]
    B --> C[调用process_text函数处理文本]
    
    C --> D[将核心观点作为专业化表达版本]
    D --> E[生成大白话版本]
    E --> F[生成理论联系实际版本]
    F --> G[生成文章摘要]
    G --> H[生成文章标题]
    H --> I[生成配图推荐]
    I --> J[组装Markdown文件内容]
    
    J --> K[保存到output.md文件]
    K --> L[结束]
    
    subgraph 生成大白话版本流程
    E1[使用professional_to_simple_template提示词] --> E2[调用LLM模型]
    E2 --> E3[过滤<think>标签内容]
    end
    
    subgraph 生成理论联系实际版本流程
    F1[使用theory_to_practice_template提示词] --> F2[调用LLM模型]
    F2 --> F3[过滤<think>标签内容]
    end
    
    subgraph 生成文章摘要流程
    G1[使用generate_summary_template提示词] --> G2[调用LLM模型]
    G2 --> G3[过滤<think>标签内容]
    end
    
    subgraph 生成文章标题流程
    H1[使用generate_title_template提示词] --> H2[调用LLM模型]
    H2 --> H3[过滤<think>标签内容]
    end
    
    subgraph 生成配图推荐流程
    I1[使用generate_image_theme_template提示词] --> I2[调用LLM模型]
    I2 --> I3[过滤<think>标签内容]
    end
    
    E --> E1
    F --> F1
    G --> G1
    H --> H1
    I --> I1
    
    subgraph 辅助函数
    M[create_prompt_chain] 
    N[filter_think_tags]
    O[save_to_markdown]
    end
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import re

# 初始化LLM模型
llm = ChatOpenAI(model="deepseek-r1", openai_api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4", base_url="https://www.gptapi.us/v1")

# 定义提示词模板
professional_to_simple_template = """
待整理的信息:
<开始>
{professional_text}
</结束>

请改写上面的待整理信息中的专业化表达，让大众更容易理解
"""

theory_to_practice_template = """
待整理的信息:
<开始>
{professional_text}
</结束>

请根据上面的待整理信息，将理论联系实际，给出对人的指导建议
"""

generate_summary_template = """
待整理的信息:
<开始>
{professional_text}
{simple_text}
{practical_text}
</结束>

请根据上面的待整理信息，生成一个简短的文章摘要给我
"""

generate_title_template = """
待整理的信息:
<开始>
{professional_text}
{simple_text}
{practical_text}
</结束>

请根据上面的待整理信息，生成一个合适的文章标题给我
"""

generate_image_theme_template = """
待整理的信息:
<开始>
{professional_text}
{simple_text}
{practical_text}
</结束>

请根据上面的待整理信息，我想给文章选一些配图，请问什么类型的主题图片合适
"""

# 创建提示词处理链
def create_prompt_chain(template):
    prompt = PromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

# 过滤掉<think>标签中的内容
def filter_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# 主处理函数
def process_text(core_idea):
    # 1. 展开核心观点，得到专业化表达版本
    professional_text = core_idea
    
    # 2. 生成大白话版本
    simple_chain = create_prompt_chain(professional_to_simple_template)
    simple_text = simple_chain.invoke({"professional_text": professional_text})
    simple_text = filter_think_tags(simple_text)
    
    # 3. 生成理论联系实际版本
    practical_chain = create_prompt_chain(theory_to_practice_template)
    practical_text = practical_chain.invoke({"professional_text": professional_text})
    practical_text = filter_think_tags(practical_text)
    
    # 4. 生成文章摘要
    summary_chain = create_prompt_chain(generate_summary_template)
    summary = summary_chain.invoke({
        "professional_text": professional_text,
        "simple_text": simple_text,
        "practical_text": practical_text
    })
    summary = filter_think_tags(summary)
    
    # 5. 生成文章标题
    title_chain = create_prompt_chain(generate_title_template)
    title = title_chain.invoke({
        "professional_text": professional_text,
        "simple_text": simple_text,
        "practical_text": practical_text
    })
    title = filter_think_tags(title)
    
    # 6. 生成配图推荐
    image_chain = create_prompt_chain(generate_image_theme_template)
    image_recommendation = image_chain.invoke({
        "professional_text": professional_text,
        "simple_text": simple_text,
        "practical_text": practical_text
    })
    image_recommendation = filter_think_tags(image_recommendation)
    
    # 7. 生成markdown文件内容
    markdown_content = f"""
# 文章标题
{title}

## 摘要
{summary}

## 专业化表达版本
{professional_text}

## 大白话版本
{simple_text}

## 理论联系实际
{practical_text}

## 配图推荐
{image_recommendation}
"""
    
    return markdown_content

# 保存到文件
def save_to_markdown(content, filename="output.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"内容已保存到 {filename}")

# 使用示例
if __name__ == "__main__":
    # 从文件读取核心观点
    try:
        with open("core_idea.txt", "r", encoding="utf-8") as f:
            core_idea = f.read().strip()
        print(f"已从文件读取核心观点，长度为{len(core_idea)}字符")
    except FileNotFoundError:
        print("未找到core_idea.txt文件，请创建该文件并在其中写入核心观点")
        exit(1)
    
    # 处理文本
    markdown_content = process_text(core_idea)
    
    # 保存到文件
    save_to_markdown(markdown_content)
    
    print("处理完成!")