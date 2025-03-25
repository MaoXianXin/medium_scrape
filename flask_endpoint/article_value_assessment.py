import requests
import os
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re
import glob
import argparse
import time

# Define the article value assessment template
assessment_template = """# 文章价值评估报告

## 文章信息
- 标题：[文章标题]
- 主题：[文章主题]

## 1. 创新性评估
- 文章提出的核心观点/技术/方法有何创新之处？
- 与现有研究/技术相比有何突破或改进？
- 创新点的实际意义和潜在影响是什么？

## 2. 实用性评估
- 文章内容能否应用于实际场景？应用门槛如何？
- 实施成本与预期收益是否合理？
- 是否提供了足够的实施指南或案例？

## 3. 完整性评估
- 文章是否全面覆盖了主题的关键方面？
- 是否客观讨论了方法/技术的局限性和缺点？
- 对替代方案的比较是否充分？

## 4. 可信度评估
- 文章的论点是否有充分的数据/实验支持？
- 作者的资质和背景如何？
- 引用的来源是否权威可靠？

## 5. 特定需求评估
- 文章内容与特定领域/问题/目标的相关性如何？
- 能否解决特定领域面临的具体挑战？
- 相比当前使用的方法/技术有何优势？

## 6. 总体价值判断
- 这篇文章的价值等级（高/中/低）？
- 最值得关注和应用的1-3个要点是什么？
- 有哪些内容需要进一步验证或补充？"""

# Create LangChain prompt template
assessment_prompt_template = PromptTemplate(
    input_variables=["article_text", "summary", "assessment_template"],
    template="""请对以下文章进行全面价值评估：

文章内容：
{article_text}

文章总结：
{summary}

请按照以下评估模板进行系统评估：
{assessment_template}

请提供一个结构化的评估报告，重点突出文章的实际价值和应用建议。评估应该专业、客观，并基于文章内容给出具体的分析。"""
)

def process_article(article_path, output_dir, processed_log_path, llm):
    """Process a single article and generate a value assessment"""
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

    assessment_chain = assessment_prompt_template | llm

    # Generate the article value assessment
    try:
        response = assessment_chain.invoke({
            "article_text": article_text,
            "summary": summary,
            "assessment_template": assessment_template
        })
        # 过滤掉<think>...</think>内容
        value_assessment = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
        print(f"Value assessment generated successfully for {article_filename}.")
    except Exception as e:
        print(f"Error generating assessment for {article_filename}: {e}")
        return False

    # Save the assessment to a file
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename from the article path
    article_basename = os.path.basename(article_path).replace(".txt", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assessment_filename = f"{article_basename}_assessment_{timestamp}.md"
    assessment_path = os.path.join(output_dir, assessment_filename)

    with open(assessment_path, "w", encoding="utf-8") as f:
        f.write(value_assessment)

    # Log the processed article
    with open(processed_log_path, "a", encoding="utf-8") as f:
        f.write(f"{article_filename}\n")

    print(f"Value assessment generated and saved to: {assessment_path}")
    print(f"Article '{article_filename}' added to processed list.")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate article value assessments')
    parser.add_argument('--input_dir', type=str, default="/home/mao/workspace/medium_scrape/111",
                        help='Directory containing article txt files')
    parser.add_argument('--output_dir', type=str, default="/home/mao/workspace/medium_scrape/assessments",
                        help='Directory to save article value assessments')
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