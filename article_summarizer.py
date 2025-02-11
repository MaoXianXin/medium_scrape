"""
flowchart TD
    A[开始] --> B[初始化 ArticleSummarizer]
    B --> C[读取文章内容]
    C --> D[调用 summarize_article 方法]
    
    D --> E{检查文章长度<br/>是否超过10000 tokens?}
    E -->|是| F[返回 None]
    E -->|否| G[创建输出目录]
    
    G --> H[调用 segment_article<br/>分段文章]
    H --> I[遍历每个文章段落]
    
    I --> J[调用 generate_summary<br/>生成每段摘要]
    J --> K[保存对话历史<br/>_save_conversation]
    
    K --> L{是否还有更多段落?}
    L -->|是| I
    L -->|否| M[合并所有摘要]
    
    M --> N[保存完整摘要到文件]
    N --> O[返回完整摘要]
    O --> P[结束]

    subgraph "segment_article 方法"
    Q[接收文章文本] --> R[将文本转换为 tokens]
    R --> S[按 max_tokens_per_segment<br/>分割文章]
    S --> T[处理最后一个分段]
    T --> U[返回分段列表]
    end

    subgraph "generate_summary 方法"
    V[创建提示模板] --> W[调用 LLM 生成摘要]
    W --> X[保存对话记录]
    X --> Y[返回摘要内容]
    end

classDiagram
    class ArticleSummarizer {
        -llm: ChatOpenAI
        -max_tokens_per_segment: int
        -tokenizer: tiktoken
        +__init__(openai_api_key, model, temperature, base_url, max_tokens_per_segment)
        +summarize_article(article, output_dir)
        -segment_article(article)
        -generate_summary(article)
        -_save_conversation(prompt, response, conversation_type)
    }
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
import tiktoken
import hashlib
import time

class ArticleSummarizer:
    def __init__(self, openai_api_key, model="gpt-4-turbo", temperature=0.3, base_url=None, max_tokens_per_segment=2000):
        """
        Initialize the ArticleSummarizer with OpenAI API key and optional parameters
        
        Args:
            openai_api_key (str): OpenAI API key for authentication
            model (str, optional): OpenAI model to use. Defaults to "gpt-4-turbo"
            temperature (float, optional): Model temperature setting. Defaults to 0.3
            base_url (str, optional): Base URL for API endpoint
            max_tokens_per_segment (int, optional): Maximum tokens per segment. Defaults to 2000
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        llm_params = {
            "model": model,
            "temperature": temperature
        }
        if base_url:
            llm_params["base_url"] = base_url
        
        self.llm = ChatOpenAI(**llm_params)
        self.max_tokens_per_segment = max_tokens_per_segment
        self.tokenizer = tiktoken.encoding_for_model("gpt-4-turbo")

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

    def summarize_article(self, article, output_dir="summary_results"):
        """
        Complete article summarization workflow
        
        Args:
            article (str): Full article text
            output_dir (str): Directory to save the summary results
            
        Returns:
            str: Complete article summary
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
        
        # Generate summary for each segment
        segment_summaries = []
        for segment in segments:
            summary = self.generate_summary(segment)
            segment_summaries.append(summary)
        
        # Combine all summaries
        full_summary = "\n\n".join(segment_summaries)
        
        # Save the summary
        with open(os.path.join(output_dir, "article_summary.txt"), "w", encoding="utf-8") as f:
            f.write(full_summary)
        
        return full_summary


# Usage example
if __name__ == "__main__":
    summarizer = ArticleSummarizer(
        openai_api_key="sk-noerGmiAt3J8SQdnj1UI74K4ixZhB55OUuEp6rfa85BOjVcI",
        model="deepseek-r1",
        temperature=0.3,
        base_url="https://zzzzapi.com/v1"  # optional
    )

    with open("article.txt", "r", encoding="utf-8") as f:
        article = f.read()

    summary = summarizer.summarize_article(article)
    print(summary)