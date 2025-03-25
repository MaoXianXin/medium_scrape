import requests
import os
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re
import glob
import argparse
import time

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

def process_article(article_path, output_dir, processed_log_path, llm):
    """Process a single article and generate a technical brief"""
    article_filename = os.path.basename(article_path)
    
    # Read article text from a file
    try:
        with open(article_path, "r", encoding="utf-8") as file:
            article_text = file.read()
    except FileNotFoundError:
        print(f"Error: article file not found: {article_path}")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

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
        print(f"Error for {article_filename}:", result.get("error", "Unknown error"))
        return False

    summary = result["summary"]
    print(f"Summary obtained successfully for {article_filename}.")

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
        print(f"Technical brief generated successfully for {article_filename}.")
    except Exception as e:
        print(f"Error generating brief for {article_filename}: {e}")
        return False

    # Save the brief to a file
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename from the article path
    article_basename = os.path.basename(article_path).replace(".txt", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    brief_filename = f"{article_basename}_brief_{timestamp}.md"
    brief_path = os.path.join(output_dir, brief_filename)

    with open(brief_path, "w", encoding="utf-8") as f:
        f.write(technical_brief)

    # Log the processed article
    with open(processed_log_path, "a", encoding="utf-8") as f:
        f.write(f"{article_filename}\n")

    print(f"Technical brief generated and saved to: {brief_path}")
    print(f"Article '{article_filename}' added to processed list.")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate technical briefs from articles')
    parser.add_argument('--input_dir', type=str, default="/home/mao/workspace/medium_scrape/111",
                        help='Directory containing article txt files')
    parser.add_argument('--output_dir', type=str, default="/home/mao/workspace/medium_scrape/briefs",
                        help='Directory to save technical briefs')
    parser.add_argument('--log_path', type=str, 
                        default="/home/mao/workspace/medium_scrape/flask_endpoint/processed_articles.txt",
                        help='Path to the processed articles log file')
    args = parser.parse_args()

    # Initialize LLM
    llm = ChatOpenAI(
        model="deepseek-r1",
        temperature=0.3,
        api_key="sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
        base_url="https://www.gptapi.us/v1"
    )

    # Get all txt files in the input directory
    article_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    
    if not article_files:
        print(f"No .txt files found in {args.input_dir}")
        return
    
    # Load processed articles log
    processed_articles = set()
    if os.path.exists(args.log_path):
        with open(args.log_path, "r", encoding="utf-8") as f:
            processed_articles = set(line.strip() for line in f.readlines())
    else:
        # Create the log file if it doesn't exist
        with open(args.log_path, "w", encoding="utf-8") as f:
            pass
    
    # Filter out already processed articles
    unprocessed_articles = [article for article in article_files 
                           if os.path.basename(article) not in processed_articles]
    
    print(f"Found {len(article_files)} article files, {len(unprocessed_articles)} need processing")
    
    # Process each unprocessed article
    for i, article_path in enumerate(unprocessed_articles):
        print(f"\nProcessing article {i+1}/{len(unprocessed_articles)}: {os.path.basename(article_path)}")
        success = process_article(article_path, args.output_dir, args.log_path, llm)
        
        # Add a small delay between API calls to avoid rate limiting
        if i < len(unprocessed_articles) - 1 and success:
            time.sleep(2)
    
    print("\nAll articles processed!")

if __name__ == "__main__":
    main()