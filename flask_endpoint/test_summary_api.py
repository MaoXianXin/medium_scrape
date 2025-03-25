import requests
import os
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re
# Read article text from a file
try:
    article_path = "/home/mao/workspace/medium_scrape/articles/4-bit-quantization-with-gptq-36b0f4f02c34.txt"
    with open(article_path, "r", encoding="utf-8") as file:
        article_text = file.read()
except FileNotFoundError:
    print(f"Error: article file not found: {article_path}")
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Get article summary
url = "http://localhost:5000/api/summarize"
headers = {"Content-Type": "application/json"}
payload = {
    "article_text": article_text,
    "temperature": 0.2,
    "model": "deepseek-r1",
    "api_key": "sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
    "base_url": "https://www.gptapi.us/v1"
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()

if response.status_code != 200:
    print("Error:", result.get("error", "Unknown error"))
    exit(1)

summary = result["summary"]
print("Summary obtained successfully.")

# Define the brief template
brief_template = """# [技术简报标题]

## 内容摘要
[2-3句话的核心内容概述]

## 关键发现
- [关键发现1]
- [关键发现2]
- [关键发现3]
- [关键发现4]

## 技术背景
[简洁的背景信息，1-2段]

## 核心技术
### [技术点1]
[技术详情，保持简洁]

### [技术点2]
[技术详情，保持简洁]

## 实际应用
[列出3-4个具体应用场景]

## 结论与展望
[总结技术价值和未来方向]

## 术语表
- [术语1]：[简短解释]
- [术语2]：[简短解释]"""

# Create LangChain prompt template
brief_prompt_template = PromptTemplate(
    input_variables=["article_text", "summary", "brief_template"],
    template="""根据以下文章内容和总结，按照给定的模板生成一份技术简报。
    
文章内容：
{article_text}

文章总结：
{summary}

简报模板：
{brief_template}

请生成一份完整的技术简报，填充模板中的所有部分。简报应该专业、简洁，并突出文章中的关键技术信息。"""
)


llm = ChatOpenAI(
    model="deepseek-r1",
    temperature=0.3,
    api_key="sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
    base_url="https://www.gptapi.us/v1"
)


brief_chain = brief_prompt_template | llm

# Generate the technical brief
try:
    response = brief_chain.invoke({
        "article_text": article_text,
        "summary": summary,
        "brief_template": brief_template
    })
    # 过滤掉<think>...</think>内容
    technical_brief = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
    print("Technical brief generated successfully.")
except Exception as e:
    print(f"Error generating brief: {e}")
    exit(1)

# Save the brief to a file
output_dir = "/home/mao/workspace/medium_scrape/briefs"
os.makedirs(output_dir, exist_ok=True)

# Extract filename from the article path
article_filename = os.path.basename(article_path).replace(".txt", "")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
brief_filename = f"{article_filename}_brief_{timestamp}.md"
brief_path = os.path.join(output_dir, brief_filename)

with open(brief_path, "w", encoding="utf-8") as f:
    f.write(technical_brief)

print(f"Technical brief generated and saved to: {brief_path}")