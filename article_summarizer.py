"""
classDiagram
    class ArticleSummarizer {
        +__init__(openai_api_key, summary_model, extraction_model, temperature, base_url, max_tokens_per_segment)
        +batch_process_articles(input_dir, output_dir) 
        -_save_conversation(prompt, response, conversation_type)
        +segment_article(article)
        +generate_summary(article)
        +extract_key_points(summary)
        +summarize_article(article, output_dir, article_path, file_prefix)
    }

    ArticleSummarizer --> batch_process_articles : 批量处理入口
    batch_process_articles --> summarize_article : 处理单篇文章
    
    summarize_article --> segment_article : 1. 文章分段
    summarize_article --> generate_summary : 2. 生成摘要
    summarize_article --> extract_key_points : 3. 提取关键点
    
    generate_summary --> _save_conversation : 保存对话历史
    extract_key_points --> _save_conversation : 保存对话历史
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
import tiktoken
import hashlib
import time

class ArticleSummarizer:
    def __init__(self, openai_api_key, summary_model="gpt-4-turbo", extraction_model="deepseek-v3", 
                 temperature=0.3, base_url=None, max_tokens_per_segment=2000):
        """
        Initialize the ArticleSummarizer with OpenAI API key and optional parameters
        
        Args:
            openai_api_key (str): OpenAI API key for authentication
            summary_model (str, optional): Model for generating summary. Defaults to "gpt-4-turbo"
            extraction_model (str, optional): Model for extracting key points. Defaults to "deepseek-v3"
            temperature (float, optional): Model temperature setting. Defaults to 0.3
            base_url (str, optional): Base URL for API endpoint
            max_tokens_per_segment (int, optional): Maximum tokens per segment. Defaults to 2000
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 初始化用于生成总结的模型
        summary_params = {
            "model": summary_model,
            "temperature": temperature
        }
        if base_url:
            summary_params["base_url"] = base_url
        
        self.llm = ChatOpenAI(**summary_params)
        
        # 初始化用于提取核心观点的模型
        extraction_params = summary_params.copy()
        extraction_params["model"] = extraction_model
        self.extraction_llm = ChatOpenAI(**extraction_params)
        
        self.max_tokens_per_segment = max_tokens_per_segment
        self.tokenizer = tiktoken.encoding_for_model("gpt-4-turbo")

    def batch_process_articles(self, input_dir, output_dir="summary_results"):
        """
        批量处理指定目录下的所有txt文件
        
        Args:
            input_dir (str): 输入文件夹路径，包含待处理的txt文件
            output_dir (str): 输出文件夹路径，用于保存处理结果
            
        Returns:
            dict: 包含所有文章处理结果的字典，格式为 {文件名: 处理结果}
        """
        if not os.path.exists(input_dir):
            print(f"输入目录 {input_dir} 不存在")
            return None
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建一个统一的核心观点文件
        all_key_points_file = os.path.join(output_dir, "all_key_points.txt")
        
        results = {}
        # 遍历目录下的所有txt文件
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                article_path = os.path.join(input_dir, filename)
                try:
                    prefix = filename[:-4]  # 移除.txt后缀
                    
                    print(f"正在处理文件: {filename}")
                    with open(article_path, 'r', encoding='utf-8') as f:
                        article = f.read()
                    
                    # 处理单个文章
                    result = self.summarize_article(
                        article, 
                        output_dir=output_dir,
                        article_path=article_path,
                        file_prefix=prefix + "_"
                    )
                    
                    if result:
                        results[filename] = result
                        # 将核心观点追加到统一文件中
                        with open(all_key_points_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n{'='*50}\n")
                            f.write(f"文件名：{filename}\n")
                            f.write(f"文件路径：{article_path}\n")
                            f.write(f"{'='*50}\n\n")
                            f.write(result['key_points'])
                            f.write("\n\n")
                    else:
                        print(f"文件 {filename} 处理失败")
                        
                except Exception as e:
                    print(f"处理文件 {filename} 时发生错误: {str(e)}")
                    results[filename] = {"error": str(e)}
                    
        return results

    def _save_conversation(self, prompt, response, conversation_type="summary"):
        """Save conversation history to a JSON file"""
        conversation_id = hashlib.md5(str(time.time()).encode()).hexdigest()
        
        conversation_history = [{
            "id": f"{conversation_type}_{conversation_id}",
            "conversations": [
                {"from": "user", "value": prompt},
                {"from": "assistant", "value": response}
            ]
        }]
        
        save_dir = "conversation_history"
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{save_dir}/conversation_{conversation_type}_{conversation_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
            
        return conversation_id

    def segment_article(self, article):
        """Segment article based on specified token count"""
        tokens = self.tokenizer.encode(article, disallowed_special=())
        segments = []
        current_segment_tokens = []
        current_length = 0
        
        for token in tokens:
            if current_length >= self.max_tokens_per_segment:
                segment_text = self.tokenizer.decode(current_segment_tokens)
                segments.append(segment_text)
                current_segment_tokens = [token]
                current_length = 1
            else:
                current_segment_tokens.append(token)
                current_length += 1
        
        if current_segment_tokens:
            if len(current_segment_tokens) >= 1000:
                segment_text = self.tokenizer.decode(current_segment_tokens)
                segments.append(segment_text)
            elif segments:
                last_segment = segments[-1]
                additional_text = self.tokenizer.decode(current_segment_tokens)
                segments[-1] = last_segment + additional_text
        
        return segments

    def generate_summary(self, article):
        """Generate a summary of the article"""
        summary_prompt = PromptTemplate(
            input_variables=["article"],
            template="""
1. 文章概述:
   - 主题:
   - 作者的写作目的:
   - 文章的整体结构:

2. 核心观点提炼:
   - 主要观点1:
   - 主要观点2:
   - [根据需要添加更多]

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

根据上述的方法，对以下文章进行全面的内容提炼和分析，请用中文回复:
---待处理的文章内容---
{article}
---待处理的文章内容---"""
        )
        
        chain = summary_prompt | self.llm
        summary = chain.invoke({"article": article}).content
        
        self._save_conversation(
            prompt=summary_prompt.format(article=article),
            response=summary,
            conversation_type="summary"
        )
        
        return summary

    def extract_key_points(self, summary):
        """
        使用 deepseek-v3 模型从总结中提取核心观点
        
        Args:
            summary (str): 文章总结内容
            
        Returns:
            str: 提取的核心观点
        """
        extraction_prompt = PromptTemplate(
            input_variables=["summary"],
            template="""
请从以下文章总结中提取"核心观点提炼"部分的内容。只需要返回核心观点部分，不需要其他内容：
---待处理的文章总结内容---
{summary}
---待处理的文章总结内容---

请按照以下格式返回：
核心观点：
1. [观点1]
2. [观点2]
...
"""
        )
        
        chain = extraction_prompt | self.extraction_llm
        key_points = chain.invoke({"summary": summary}).content
        
        self._save_conversation(
            prompt=extraction_prompt.format(summary=summary),
            response=key_points,
            conversation_type="key_points_extraction"
        )
        
        return key_points

    def summarize_article(self, article, output_dir="summary_results", article_path=None, file_prefix=""):
        """
        Complete article summarization workflow
        
        Args:
            article (str): Full article text
            output_dir (str): Directory to save the summary results
            article_path (str, optional): Path of the source article
            file_prefix (str): Prefix for output files
            
        Returns:
            dict: Complete article summary and key points
        """
        # Check token count
        tokens = self.tokenizer.encode(article, disallowed_special=())
        token_count = len(tokens)
        print(f"Article token count: {token_count}")
        
        if token_count > 10000:
            print(f"Article is too long ({token_count} tokens). Maximum allowed is 10,000 tokens.")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Segment article if necessary
        segments = self.segment_article(article)
        
        # Generate summary and extract key points for each segment
        segment_summaries = []
        segment_key_points = []
        for i, segment in enumerate(segments):
            summary = self.generate_summary(segment)
            key_points = self.extract_key_points(summary)
            
            segment_summaries.append(summary)
            segment_key_points.append(key_points)
            
            # Save individual segment results if multiple segments
            if len(segments) > 1:
                segment_file = os.path.join(output_dir, f"{file_prefix}segment_{i+1}_summary.txt")
                with open(segment_file, "w", encoding="utf-8") as f:
                    f.write(summary)
                
                segment_key_points_file = os.path.join(output_dir, f"{file_prefix}segment_{i+1}_key_points.txt")
                with open(segment_key_points_file, "w", encoding="utf-8") as f:
                    if article_path:
                        f.write(f"来源文章段落 {i+1}：{article_path}\n\n")
                    f.write(key_points)
        
        # Combine all summaries and key points
        full_summary = "\n\n".join(segment_summaries)
        all_key_points = "\n\n".join(segment_key_points)
        
        # Save the complete summary and key points
        summary_file = os.path.join(output_dir, f"{file_prefix}article_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(full_summary)
        
        key_points_file = os.path.join(output_dir, f"{file_prefix}key_points.txt")
        with open(key_points_file, "w", encoding="utf-8") as f:
            if article_path:
                f.write(f"来源文章：{article_path}\n\n")
            f.write(all_key_points)
        
        return {
            "full_summary": full_summary,
            "key_points": all_key_points
        }


# Usage example
if __name__ == "__main__":
    summarizer = ArticleSummarizer(
        openai_api_key="sk-jsbIUYq1MwwUheWDDeB01c1a7a2246E3956b6b75762c9f21",
        summary_model="deepseek-r1",      # 用于生成总结的模型
        extraction_model="deepseek-v3",   # 用于提取核心观点的模型
        temperature=0.3,
        base_url="https://www.gptapi.us/v1"  # optional
    )

    # 支持单文件处理
    article_path = "/home/mao/workspace/medium_scrape/articles/meta-releases-llama-3-heres-all-you-need-to-know-88d850cabedd.txt"
    if os.path.isfile(article_path):
        with open(article_path, "r", encoding="utf-8") as f:
            article = f.read()
        summary = summarizer.summarize_article(article, article_path=article_path)
        print(summary)
    
    # 支持批量处理
    input_directory = "articles"  # 包含多个txt文件的目录
    if os.path.isdir(input_directory):
        results = summarizer.batch_process_articles(input_directory)
        print("\n批量处理结果:")
        for filename, result in results.items():
            print(f"\n文件: {filename}")
            print(result)