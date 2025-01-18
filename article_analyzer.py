# Article Summary and Core Points Extraction
"""
flowchart TD
    A["原始文章 
    (Original Article)"] --> B["文章总结提示词 
    (Summary Prompt)"]
    B --> C["文章总结 
    (Article Summary)"]
    C --> D["核心观点提取提示词 
    (Core Points Extraction Prompt)"]
    D --> E["核心观点列表 
    (Core Points List)"]
    
    A & E --> F["详细分析提示词 
    (Detailed Analysis Prompt)"]
    F --> G["每个核心观点的详细分析 
    (Detailed Analysis for Each Point)"]
    
    G --> H["标题和摘要生成提示词
    (Title and Brief Generation Prompt)"]
    H --> I["标题和文章信息摘要
    (Title and Article Brief)"]
    
    I & C & E & G --> J["保存分析结果 
    (Save Analysis Results)"]
    
    subgraph Output["输出文件 (Output Files)"]
        K["summary.txt"]
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

class ArticleAnalyzer:
    def __init__(self, openai_api_key, model="gpt-4-turbo", temperature=0.3, base_url=None):
        """
        Initialize the ArticleAnalyzer with OpenAI API key and optional parameters
        
        Args:
            openai_api_key (str): OpenAI API key for authentication
            model (str, optional): OpenAI model to use. Defaults to "gpt-4-turbo"
            temperature (float, optional): Model temperature setting. Defaults to 0.3
            base_url (str, optional): Base URL for API endpoint
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        llm_params = {
            "model": model,
            "temperature": temperature
        }
        if base_url:
            llm_params["base_url"] = base_url
        
        self.llm = ChatOpenAI(**llm_params)

    def generate_summary(self, article):
        """
        Generate a summary of the article
        
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
        summary = chain.invoke({"article": article}).content
        return summary

    def extract_core_points(self, summary):
        """
        Extract core points from the summary in JSON format
        
        Args:
            summary (str): Summarized article
        
        Returns:
            dict: Core points in JSON format
        """
        core_points_prompt = PromptTemplate(
            input_variables=["summary"],
            template="""请从以下文章总结中，仅提取"核心观点提炼"部分的内容。

要求:
1. 只关注"核心观点提炼"部分的内容
2. 将每个主要观点转换为JSON格式
3. 使用以下JSON格式返回：
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
        
        try:
            core_points = json.loads(core_points_str)
            return core_points.get('points', [])
        except json.JSONDecodeError:
            core_points_str = core_points_str.split('```json')[-1].split('```')[0].strip()
            try:
                core_points = json.loads(core_points_str)
                return core_points.get('points', [])
            except:
                return []

    def extract_detailed_points(self, article, core_points):
        """
        Extract detailed information for each core point
        
        Args:
            article (str): Full article text
            core_points (list): List of core points as strings
        
        Returns:
            dict: Detailed information for each core point
        """
        detailed_points = {}
        
        detailed_prompt = PromptTemplate(
            input_variables=["article", "point"],
            template="""基于原文，详细阐述以下核心观点:

核心观点: {point}

要求:
1. 从原文中提取支持该观点的具体证据
2. 解释观点的深层含义
3. 分析观点的实际意义和潜在影响
4. 保持客观和专业的分析态度

原文:
{article}

详细分析:"""
        )
        
        detailed_chain = detailed_prompt | self.llm
        
        for point in core_points:
            detailed_analysis = detailed_chain.invoke({
                "article": article,
                "point": point
            }).content
            detailed_points[point] = detailed_analysis
        
        return detailed_points

    def generate_title_and_brief(self, detailed_analysis):
        """
        基于详细分析生成文章标题和简要信息摘要
        
        Args:
            detailed_analysis (dict): 详细分析内容
            
        Returns:
            tuple: (标题, 简要摘要)
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
        
        return title.strip(), brief.strip()

    def analyze_article(self, article):
        """
        Complete article analysis workflow
        
        Args:
            article (str): Full article text
        
        Returns:
            dict: Comprehensive article analysis
        """
        # 1. 生成文章摘要
        summary = self.generate_summary(article)
        
        # 2. 提取核心观点
        core_points = self.extract_core_points(summary)
        
        # 3. 提取详细信息
        detailed_points = self.extract_detailed_points(article, core_points)
        
        # 4. 保存结果到文件
        output_dir = "analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存摘要
        with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)
        
        # 保存核心观点
        with open(os.path.join(output_dir, "core_points.txt"), "w", encoding="utf-8") as f:
            for point in core_points:
                f.write(f"{point}\n")
                f.write("-" * 50 + "\n")
        
        # 保存详细分析
        with open(os.path.join(output_dir, "detailed_points.txt"), "w", encoding="utf-8") as f:
            for point, analysis in detailed_points.items():
                f.write(f"核心观点：{point}\n")
                f.write("详细分析：\n")
                f.write(analysis)
                f.write("\n" + "=" * 50 + "\n\n")
        
        # 生成标题和简要摘要
        title, brief = self.generate_title_and_brief(detailed_points)
        
        # 保存完整内容到单个文件
        with open(os.path.join(output_dir, "complete_analysis.txt"), "w", encoding="utf-8") as f:
            f.write(f"标题：{title}\n\n")
            f.write("文章信息摘要：\n")
            f.write(f"{brief}\n\n")
            f.write("=" * 50 + "\n\n")
            f.write("详细分析：\n")
            for point, analysis in detailed_points.items():
                f.write(f"核心观点：{point}\n")
                f.write("详细分析：\n")
                f.write(analysis)
                f.write("\n" + "=" * 50 + "\n\n")
        
        return {
            'title': title,
            'brief': brief,
            'summary': summary,
            'core_points': core_points,
            'detailed_points': detailed_points
        }

# 使用示例
def main():
    openai_api_key = "sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4"
    base_url = "https://www.gptapi.us/v1"  # 可选
    analyzer = ArticleAnalyzer(
        openai_api_key,
        model="claude-3-5-sonnet",
        temperature=0.5,
        base_url=base_url
    )
    
    # 从文件读取文章内容
    file_path = "/home/mao/workspace/medium_scrape/articles/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"  # 指定文件路径
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            article = file.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return
    
    if not article.strip():
        print("错误：文件内容为空")
        return
    
    result = analyzer.analyze_article(article)
    
    # 打印结果
    print("文章摘要:", result['summary'])
    print("\n核心观点:")
    for point in result['core_points']:
        print(f"- {point}")
    
    print("\n详细分析:")
    for point, analysis in result['detailed_points'].items():
        print(f"\n核心观点: {point}")
        print(f"详细分析: {analysis}")

if __name__ == "__main__":
    main()