# Article Summary and Core Points Extraction
"""
flowchart TD
    A["原始文章 
    (Original Article)"] --> T1["检查文章Tokens
    (Check Article Tokens)"]
    
    T1 -- "Tokens > 10000" --> T2["跳过分析
    (Skip Analysis)"]
    T1 -- "Tokens ≤ 10000" --> O["文章分段 
    (Article Segmentation)"]
    
    O --> P["按Tokens=2000切分 
    (Segment by 2000 Tokens)"]
    
    subgraph Segmentation["分段处理 (Segment Processing)"]
        P --> Q["分段文章1 
        (Segment 1)"]
        P --> R["分段文章2 
        (Segment 2)"]
        P --> S["分段文章N 
        (Segment N)"]
    end
    
    subgraph SummaryProcess["总结处理流程 (Summary Process)"]
        Q --> B1["文章总结提示词 
        (Summary Prompt 1)"]
        B1 --> C1["文章总结1 
        (Article Summary 1)"]
        C1 --> D1["核心观点提取提示词 
        (Core Points Extraction Prompt 1)"]
        D1 --> E1["核心观点列表1 
        (Core Points List 1)"]
        
        R --> B2["文章总结提示词 
        (Summary Prompt 2)"]
        B2 --> C2["文章总结2 
        (Article Summary 2)"]
        C2 --> D2["核心观点提取提示词 
        (Core Points Extraction Prompt 2)"]
        D2 --> E2["核心观点列表2 
        (Core Points List 2)"]
        
        S --> BN["文章总结提示词 
        (Summary Prompt N)"]
        BN --> CN["文章总结N 
        (Article Summary N)"]
        CN --> DN["核心观点提取提示词 
        (Core Points Extraction Prompt N)"]
        DN --> EN["核心观点列表N 
        (Core Points List N)"]
    end
    
    C1 & C2 & CN --> X["完整文章总结 
    (Comprehensive Article Summary)"]
    
    E1 & E2 & EN --> T["观点整理提示词 
    (Core Points Consolidation Prompt)"]
    T --> U["最终核心观点列表 
    (Final Consolidated Core Points List)"]
    
    A & U --> F["详细分析提示词 
    (Detailed Analysis Prompt)"]
    F --> G["每个核心观点的详细分析 
    (Detailed Analysis for Each Point)"]
    
    G --> H["标题和摘要生成提示词
    (Title and Brief Generation Prompt)"]
    H --> I["标题和文章信息摘要
    (Title and Article Brief)"]
    
    I & X & U & G --> J["保存分析结果 
    (Save Analysis Results)"]
    
    subgraph Output["输出文件 (Output Files)"]
        K["summary.txt: 文章总结1 + 文章总结2 + ... + 文章总结N 
        (Concatenated Summaries)"]
        L["core_points.txt"]
        M["detailed_points.txt"]
        N["complete_analysis.txt"]
    end
    
    J --> K
    J --> L
    J --> M
    J --> N
"""
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
import tiktoken
import hashlib
import time
import random

class ArticleAnalyzer:
    def __init__(self, openai_api_key, model="gpt-4-turbo", temperature=0.3, base_url=None, max_tokens_per_segment=2000):
        """
        Initialize the ArticleAnalyzer with OpenAI API key and optional parameters
        
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
        """
        Save conversation history to a JSON file
        
        Args:
            prompt (str): The prompt sent to the model
            response (str): The model's response
            conversation_type (str, optional): Type of conversation for reference. Defaults to "summary"
            
        Returns:
            str: The conversation ID
        """
        # 生成随机hash值作为对话id
        conversation_id = hashlib.md5(str(time.time()).encode()).hexdigest()
        
        # 构建对话记录
        conversation_history = [{
            "id": f"{conversation_type}_{conversation_id}",
            "conversations": [
                {
                    "from": "user",
                    "value": prompt
                },
                {
                    "from": "assistant",
                    "value": response
                }
            ]
        }]
        
        # 确保保存目录存在
        save_dir = "conversation_history"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存对话记录到json文件
        filename = f"{save_dir}/conversation_{conversation_type}_{conversation_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
            
        return conversation_id

    def _clean_json_string(self, json_str):
        # 移除可能的 JSON 代码块标记
        if '```json' in json_str:
            json_str = json_str.split('```json')[-1].split('```')[0]
        return json_str.strip()
    
    def generate_summary(self, article):
        """
        Generate a summary of the article and save the conversation history
        
        Args:
            article (str): Full text of the article
        
        Returns:
            str: Summarized article
        """
        summary_prompt = PromptTemplate(
            input_variables=["article"],
            template="""请对以下文章进行专业、系统的分析总结：

1. 文章概述:
   - 主题:
   - 作者的写作目的:
   - 文章的整体结构:

2. 核心观点提炼:
   - 主要观点1:
   - 主要观点2:
   [根据内容添加更多观点]

3. 论据分析:
   对于每个核心观点:
   - 支持论据:
   - 论据类型(数据/案例/专家观点等):
   - 论据的有效性评估:

4. 副观点和支持性细节:
   - 副观点1:
   - 副观点2:
   - 重要细节:

5. 关键词和重复概念:
   [列出文章中反复出现或强调的关键词和概念]

文章内容:
{article}

请按照上述结构进行分析："""
        )
        
        # 使用新的方式创建和调用链
        chain = summary_prompt | self.llm
        
        # 获取summary
        summary = chain.invoke({"article": article}).content
        
        # 保存对话记录
        # self._save_conversation(
        #     prompt=summary_prompt.format(article=article),
        #     response=summary,
        #     conversation_type="summary"
        # )
        
        return summary

    def extract_core_points(self, summary):
        """
        Extract core points from the summary in JSON format and save the conversation history
        
        Args:
            summary (str): Summarized article
        
        Returns:
            list: List of core points extracted from the summary. Returns empty list if extraction fails or no points found.
        """
        core_points_prompt = PromptTemplate(
            input_variables=["summary"],
            template="""请从以下文章总结中，仅提取"核心观点提炼"部分的内容。

要求:
1. 只关注"核心观点提炼"部分的内容
2. 将每个主要观点转换为JSON格式
3. 核心观点内容中不能包含双引号，如有双引号请替换为单引号
4. 使用以下JSON格式返回：
{{
    "points": [
        "第一个核心观点的具体内容",
        "第二个核心观点的具体内容"
    ]
}}

文章总结:
{summary}

请仅返回JSON格式的核心观点："""
        )
        
        chain = core_points_prompt | self.llm
        core_points_str = chain.invoke({"summary": summary}).content
        print("原始输出:", core_points_str)
        
        try:
            cleaned_json = self._clean_json_string(core_points_str)
            print("清理后的 JSON:", cleaned_json)
            core_points = json.loads(cleaned_json)
            print("JSON 解析成功:", core_points)
            # 只有在成功解析JSON后才保存对话记录
            # self._save_conversation(
            #     prompt=core_points_prompt.format(summary=summary),
            #     response=core_points_str,
            #     conversation_type="core_points"
            # )
            return core_points.get('points', [])
        except json.JSONDecodeError as e:
            print("JSON 解析失败:", str(e))
            return []

    def extract_detailed_points(self, article, core_points):
        """
        Extract detailed information for each core point and save the conversation history
        
        Args:
            article (str): Full article text
            core_points (list): List of core points as strings
        
        Returns:
            dict: Detailed information for each core point, where:
                - keys are the core points (str)
                - values are the detailed analysis (str) for each point
        """
        detailed_points = {}
        
        detailed_prompt = PromptTemplate(
            input_variables=["article", "point"],
            template="""原文:
================
{article}
================

展开讲讲这点: {point}
"""
        )
        
        detailed_chain = detailed_prompt | self.llm
        
        for point in core_points:
            detailed_analysis = detailed_chain.invoke({
                "article": article,
                "point": point
            }).content
            
            # 保存对话记录
            # self._save_conversation(
            #     prompt=detailed_prompt.format(article=article, point=point),
            #     response=detailed_analysis,
            #     conversation_type="detailed_analysis"
            # )
            
            detailed_points[point] = detailed_analysis
        
        return detailed_points

    def generate_title_and_brief(self, detailed_analysis):
        """
        Generate article title and brief summary based on detailed analysis and save the conversation history
        
        Args:
            detailed_analysis (dict): Detailed analysis content
            
        Returns:
            tuple: (title: str, brief: str)
        """
        title_prompt = PromptTemplate(
            input_variables=["analysis"],
            template="""基于以下详细分析内容，请生成一个简洁、吸引人且准确的文章标题（不超过20个字）：

详细分析内容：
{analysis}

请直接返回标题（不需要任何额外说明）："""
        )
        
        brief_prompt = PromptTemplate(
            input_variables=["analysis"],
            template="""基于以下详细分析内容，请生成一段简短的文章信息摘要（100-150字），概括文章的主要内容和价值：

详细分析内容：
{analysis}

请直接返回摘要（不需要任何额外说明）："""
        )
        
        # 将所有详细分析合并为一个字符串
        full_analysis = "\n".join([f"{point}\n{analysis}" for point, analysis in detailed_analysis.items()])
        
        # 生成标题和摘要
        title_chain = title_prompt | self.llm
        brief_chain = brief_prompt | self.llm
        
        title = title_chain.invoke({"analysis": full_analysis}).content
        brief = brief_chain.invoke({"analysis": full_analysis}).content
        
        # 保存标题生成的对话记录
        # self._save_conversation(
        #     prompt=title_prompt.format(analysis=full_analysis),
        #     response=title,
        #     conversation_type="title"
        # )
        
        # 保存摘要生成的对话记录
        # self._save_conversation(
        #     prompt=brief_prompt.format(analysis=full_analysis),
        #     response=brief,
        #     conversation_type="brief"
        # )
        
        return title.strip(), brief.strip()

    def segment_article(self, article):
        """
        Segment article based on specified token count
        
        Args:
            article (str): Full article text
        
        Returns:
            list: List of article segments
        """
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
        
        # 如果最后剩余的tokens少于1000，将其合并到上一个段落
        if current_segment_tokens:
            if len(current_segment_tokens) >= 1000:
                segment_text = self.tokenizer.decode(current_segment_tokens)
                segments.append(segment_text)
            elif segments:  # 如果有前一个段落，合并到前一个段落
                last_segment = segments[-1]
                additional_text = self.tokenizer.decode(current_segment_tokens)
                segments[-1] = last_segment + additional_text
        
        return segments

    def consolidate_core_points(self, all_core_points):
        """
        Consolidate core points from all segments and save the conversation history
        
        Args:
            all_core_points (list): List of core points from all segments
        
        Returns:
            list: Consolidated list of core points. Returns empty list if consolidation fails or no points found.
        """
        consolidation_prompt = PromptTemplate(
            input_variables=["points"],
            template="""请从以下维度对文章片段中的核心观点进行系统整合：

1. 主题聚类
   - 识别相似主题的观点
   - 合并重复或高度相似的内容
   - 保留最完整、表达最准确的版本

2. 逻辑层级
   - 区分主要观点和次要观点
   - 识别观点间的因果关系
   - 建立观点的逻辑框架

3. 信息完整性
   - 确保每个观点的完整表达
   - 补充必要的上下文信息
   - 避免信息的重复和冗余

4. 表达一致性
   - 统一观点的表达方式
   - 保持语言风格的一致
   - 确保术语使用的统一

5. 重要性排序
   - 根据观点的重要程度排序
   - 突出核心价值主张
   - 保持观点的优先级

待整合的观点列表：
{points}

请按照以上维度整合观点，并使用以下JSON格式返回：
{{
    "points": [
        "整合后的观点1",
        "整合后的观点2"
    ]
}}

注意：
1. 返回的观点必须清晰、准确、完整
2. 每个观点应该独立成文
3. 严格遵守JSON格式要求"""
        )
        print("整合前的核心观点:", all_core_points)
        points_str = "\n".join([f"- {point}" for sublist in all_core_points for point in sublist])
        chain = consolidation_prompt | self.llm
        result = chain.invoke({"points": points_str}).content
        
        print("原始输出:", result)
        try:
            cleaned_json = self._clean_json_string(result)
            print("清理后的 JSON:", cleaned_json)
            consolidated = json.loads(cleaned_json)
            print("JSON 解析成功:", consolidated)
            # 只有在成功解析JSON后才保存对话记录
            # self._save_conversation(
            #     prompt=consolidation_prompt.format(points=points_str),
            #     response=result,
            #     conversation_type="consolidation"
            # )
            return consolidated.get('points', [])
        except json.JSONDecodeError as e:
            print("JSON 解析失败:", str(e))
            return []

    def analyze_article(self, article, file_prefix="", output_dir="analysis_results"):
        """
        Complete article analysis workflow
        
        Args:
            article (str): Full article text
            file_prefix (str): Prefix for output files
        
        Returns:
            dict: Comprehensive article analysis containing:
                - title (str): Article title
                - brief (str): Brief summary
                - summary (str): Full article summary
                - core_points (list): List of core points
                - detailed_points (dict): Detailed analysis for each core point
        """
        # Check token count
        tokens = self.tokenizer.encode(article, disallowed_special=())
        token_count = len(tokens)
        print(f"Article token count: {token_count}")
        
        if token_count > 10000:
            print(f"Article is too long ({token_count} tokens). Maximum allowed is 10,000 tokens.")
            return None
        
        if token_count < 2000:
            print(f"Article is too short ({token_count} tokens). Minimum required is 2,000 tokens.")
            return None
        
        # 1. 文章分段
        segments = self.segment_article(article)
        
        # 2. 对每个分段进行分析
        segment_summaries = []
        segment_core_points = []
        
        for segment in segments:
            # 生成每个分段的摘要
            summary = self.generate_summary(segment)
            segment_summaries.append(summary)
            
            # 提取每个分段的核心观点
            core_points = self.extract_core_points(summary)
            segment_core_points.append(core_points)
        
        # 3. 合并所有分段的摘要
        full_summary = "\n\n".join(segment_summaries)
        
        # 4. 整合所有分段的核心观点
        consolidated_points = self.consolidate_core_points(segment_core_points)
        print("整合后的核心观点:", consolidated_points)
        
        # 5. 对整合后的核心观点进行详细分析
        detailed_points = self.extract_detailed_points(article, consolidated_points)
        
        # 6. 生成标题和简要摘要
        title, brief = self.generate_title_and_brief(detailed_points)
        
        # 7. 保存结果到文件
        # 添加文件前缀
        prefix = f"{file_prefix}_" if file_prefix else ""
        
        # 保存摘要
        # with open(os.path.join(output_dir, f"{prefix}summary.txt"), "w", encoding="utf-8") as f:
        #     f.write(full_summary)
        
        # 保存核心观点
        # with open(os.path.join(output_dir, f"{prefix}core_points.txt"), "w", encoding="utf-8") as f:
        #     for point in consolidated_points:
        #         f.write(f"{point}\n")
        #         f.write("-" * 50 + "\n")
        
        # 保存详细分析
        # with open(os.path.join(output_dir, f"{prefix}detailed_points.txt"), "w", encoding="utf-8") as f:
        #     for point, analysis in detailed_points.items():
        #         f.write(f"核心观点：{point}\n")
        #         f.write("详细分析：\n")
        #         f.write(analysis)
        #         f.write("\n\n" + "=" * 50 + "\n\n")
        
        # 随机选择一个图片URL
        image_urls = [
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/test_imgs/2025-03-03_14-45.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306092044.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306092152.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306092255.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306101020.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306101128.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306101228.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306101358.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306101447.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306141735.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306141825.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306141927.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306142001.png",
            "https://maoxianxin1996.oss-cn-huhehaote.aliyuncs.com/ai/20250306142306.png"
        ]
        
        random_image = random.choice(image_urls)
        
        # 保存为Markdown格式
        with open(os.path.join(output_dir, f"{prefix}complete_analysis.md"), "w", encoding="utf-8") as f:
            # 添加frontmatter
            f.write("---\n")
            f.write(f"title: {title}\n")
            f.write("tags: [人工智能]\n")
            f.write(f"description: {brief}\n")
            f.write(f"image: {random_image}\n")
            f.write("---\n\n")
            
            # 添加标题和摘要
            f.write(f"标题：{title}\n\n")
            f.write("文章信息摘要：\n")
            f.write(f"{brief}\n\n")
            f.write("=" * 50 + "\n\n")
            f.write("详细分析：\n")
            for point, analysis in detailed_points.items():
                f.write(f"核心观点：{point}\n")
                f.write("详细分析：\n")
                f.write(analysis)
                f.write("\n\n" + "=" * 50 + "\n\n")
        
        # 保存原始txt格式（如果仍需要）
        with open(os.path.join(output_dir, f"{prefix}complete_analysis.txt"), "w", encoding="utf-8") as f:
            f.write(f"标题：{title}\n\n")
            f.write("文章信息摘要：\n")
            f.write(f"{brief}\n\n")
            f.write("=" * 50 + "\n\n")
            f.write("详细分析：\n")
            for point, analysis in detailed_points.items():
                f.write(f"核心观点：{point}\n")
                f.write("详细分析：\n")
                f.write(analysis)
                f.write("\n\n" + "=" * 50 + "\n\n")
        
        return {
            'title': title,
            'brief': brief,
            'summary': full_summary,
            'core_points': consolidated_points,
            'detailed_points': detailed_points
        }

# 使用示例
def main():
    openai_api_key = "sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4"
    base_url = "https://api.gptapi.us/v1"  # 可选
    analyzer = ArticleAnalyzer(
        openai_api_key,
        model="deepseek-v3",
        temperature=0.5,
        base_url=base_url
    )
    
    # 获取文件夹路径
    folder_path = "/home/mao/workspace/medium_scrape/articles"
    if not os.path.exists(folder_path):
        print(f"错误：找不到文件夹 '{folder_path}'")
        return
    
    # 获取文件夹中所有的txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"错误：在文件夹 '{folder_path}' 中没有找到txt文件")
        return
    
    # 检查分析结果目录
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理记录文件路径
    processed_record_file = "processed_files.txt"
    
    # 读取已处理文件记录
    processed_files = set()
    if os.path.exists(processed_record_file):
        with open(processed_record_file, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f if line.strip())
    
    # 处理每个txt文件
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        file_prefix = os.path.splitext(file_name)[0]
        
        # 检查是否已经分析过
        if file_name in processed_files:
            print(f"\n文件 {file_name} 已经分析过，跳过处理")
            continue
            
        print(f"\n正在处理文件: {file_name}")
        print("-" * 50)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                article = file.read()
        except Exception as e:
            print(f"读取文件 '{file_name}' 时发生错误：{str(e)}")
            continue
        
        if not article.strip():
            print(f"跳过空文件: {file_name}")
            continue
        
        result = analyzer.analyze_article(article, file_prefix, output_dir)
        
        if result is None:
            print(f"由于文章长度问题跳过分析: {file_name}")
            continue
        
        # 将处理过的文件添加到记录中
        with open(processed_record_file, 'a', encoding='utf-8') as f:
            f.write(f"{file_name}\n")
        
        # 打印结果
        print("\n文章摘要:", result['summary'])
        print("\n核心观点:")
        for point in result['core_points']:
            print(f"- {point}")
        
        print("\n详细分析:")
        for point, analysis in result['detailed_points'].items():
            print(f"\n核心观点: {point}")
            print(f"详细分析: {analysis}")
        
        print("\n" + "="*50 + "\n")  # 添加分隔线

if __name__ == "__main__":
    main()