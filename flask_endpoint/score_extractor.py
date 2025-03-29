from dialog_module.base import OneTimeDialogModule
from dialog_module.utils import create_custom_llm
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 定义评估分数提取模型
class AssessmentScores(BaseModel):
    innovation_score: int = Field(description="创新性得分")
    practicality_score: int = Field(description="实用性得分")
    completeness_score: int = Field(description="完整性得分")
    credibility_score: int = Field(description="可信度得分")
    specific_needs_score: int = Field(description="特定需求得分")
    total_score: int = Field(description="最终总分")

def extract_assessment_scores(assessment_report):
    """
    从评估报告中提取各个维度的得分
    
    参数:
        assessment_report: 文章评估报告文本
        
    返回:
        AssessmentScores对象，包含各个维度的得分
    """
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
    
    # 使用自定义模型服务
    custom_llm = create_custom_llm()
    
    # 创建评分提取模块实例
    score_extractor = OneTimeDialogModule(
        llm=custom_llm,
        prompt_template=prompt.template,
        template_variables={
            "assessment_report": assessment_report,
            "format_instructions": parser.get_format_instructions()
        }
    )
    
    # 提取评分
    extraction_result = score_extractor.process()
    
    # 解析结果为AssessmentScores对象
    try:
        scores = parser.parse(extraction_result)
        return scores
    except Exception as e:
        print(f"解析评分时出错: {e}")
        return None 