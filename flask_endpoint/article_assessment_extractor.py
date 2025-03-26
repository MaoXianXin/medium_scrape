from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import csv
import os
from pathlib import Path

# 定义输出结构
class AssessmentScores(BaseModel):
    innovation_score: int = Field(description="创新性得分")
    practicality_score: int = Field(description="实用性得分")
    completeness_score: int = Field(description="完整性得分")
    credibility_score: int = Field(description="可信度得分")
    specific_needs_score: int = Field(description="特定需求得分")
    total_score: int = Field(description="最终总分")

# 创建输出解析器
parser = PydanticOutputParser(pydantic_object=AssessmentScores)

# 定义提取提示模板
template = """
你是一个专业的文本分析助手。请从以下文章评估报告中提取各个维度的得分和最终总分。

评估报告:
{assessment_report}

请提取以下信息:
1. 创新性得分
2. 实用性得分
3. 完整性得分
4. 可信度得分
5. 特定需求得分
6. 最终总分

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["assessment_report"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 使用LLM提取信息
def extract_scores(assessment_report: str) -> AssessmentScores:
    # 初始化LLM
    llm = ChatOpenAI(
        model="deepseek-r1",
        temperature=0.3,
        api_key="sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
        base_url="https://www.gptapi.us/v1"
    )
    
    # 构建提示
    formatted_prompt = prompt.format(assessment_report=assessment_report)
    
    # 获取LLM响应
    response = llm.invoke(formatted_prompt)
    
    # 解析结果
    scores = parser.parse(response.content)
    return scores

# 使用示例
def main():
    # 获取文件夹路径
    assessment_dir = "/home/mao/workspace/medium_scrape/assessments"
    
    # CSV文件路径
    csv_file = "/home/mao/workspace/medium_scrape/assessment_scores.csv"
    file_exists = os.path.isfile(csv_file)
    
    # 检查文件是否已处理过
    processed_files = set()
    if file_exists:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            processed_files = {row['file_name'] for row in reader}
    
    # 获取所有评估报告文件
    all_assessment_files = [f for f in Path(assessment_dir).glob("*_assessment_*.md")]
    
    # 过滤出未处理的文件
    assessment_files = [f for f in all_assessment_files if f.stem not in processed_files]
    
    # 统计信息
    total_files = len(all_assessment_files)
    to_process = len(assessment_files)
    skipped_count = total_files - to_process
    processed_count = 0
    
    print(f"找到 {total_files} 个评估报告文件")
    print(f"已处理过 {skipped_count} 个文件，将跳过")
    print(f"待处理 {to_process} 个文件")
    print("-" * 50)
    
    for file_path in assessment_files:
        file_name = file_path.stem
        
        try:
            # 从文件读取评估报告
            with open(file_path, "r", encoding="utf-8") as f:
                assessment_report = f.read()
            
            # 提取得分
            scores = extract_scores(assessment_report)
            
            # 打印结果
            print(f"处理文件: {file_name}")
            print(f"  创新性得分: {scores.innovation_score}")
            print(f"  实用性得分: {scores.practicality_score}")
            print(f"  完整性得分: {scores.completeness_score}")
            print(f"  可信度得分: {scores.credibility_score}")
            print(f"  特定需求得分: {scores.specific_needs_score}")
            print(f"  最终总分: {scores.total_score}")
            
            # 将结果写入CSV文件
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = ['file_name', 'innovation_score', 'practicality_score', 'completeness_score', 
                             'credibility_score', 'specific_needs_score', 'total_score']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                
                # 写入数据
                writer.writerow({
                    'file_name': file_name,
                    'innovation_score': scores.innovation_score,
                    'practicality_score': scores.practicality_score,
                    'completeness_score': scores.completeness_score,
                    'credibility_score': scores.credibility_score,
                    'specific_needs_score': scores.specific_needs_score,
                    'total_score': scores.total_score
                })
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
    
    print(f"\n处理完成:")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_count}")
    print(f"已跳过: {skipped_count}")
    print(f"失败: {to_process - processed_count}")
    print(f"评估得分已写入 {csv_file}")

if __name__ == "__main__":
    main()