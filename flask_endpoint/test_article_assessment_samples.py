from dialog_module.base import OneTimeDialogModule
from dialog_module.utils import read_template_from_file, read_article_from_file, create_custom_llm
from test_article_summary_samples import generate_article_summary
from score_extractor import extract_assessment_scores

# 定义文章价值评估模板路径
ASSESSMENT_TEMPLATE_PATH = "templates/article_assessment_template.txt"
ASSESSMENT_PROMPT_PATH = "templates/article_assessment_prompt.txt"

def generate_article_assessment(article_text, summary=None, template_path=None):
    """
    生成文章价值评估的函数
    
    参数:
        article_text: 文章内容文本
        summary: 文章总结，如果为None则不包含总结信息
        template_path: 提示词模板路径，如果为None则使用默认模板
        
    返回:
        生成的文章价值评估报告
    """
    # 使用自定义模型服务
    custom_llm = create_custom_llm()
    
    # 读取评估模板
    assessment_template = read_template_from_file(ASSESSMENT_TEMPLATE_PATH)
    
    # 如果提供了模板路径，则从文件读取提示词模板
    if template_path is not None:
        prompt_template = read_template_from_file(template_path)
    else:
        # 否则使用默认的评估提示词模板
        prompt_template = read_template_from_file(ASSESSMENT_PROMPT_PATH)
    
    # 准备模板变量
    template_variables = {
        "article_text": article_text,
        "assessment_template": assessment_template
    }
    
    # 如果提供了总结，则添加到变量中
    if summary:
        template_variables["summary"] = summary
    else:
        # 如果没有提供总结，修改模板以移除总结部分
        prompt_template = prompt_template.replace("文章总结：\n{summary}\n\n", "")
    
    # 创建文章评估模块实例
    article_assessor = OneTimeDialogModule(
        llm=custom_llm,
        prompt_template=prompt_template,
        template_variables=template_variables
    )
    
    # 生成文章评估
    assessment = article_assessor.process()
    
    return assessment

# 示例使用
if __name__ == "__main__":
    # 指定文件路径
    file_path = "/home/mao/workspace/medium_scrape/articles/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"
    
    # 读取文章内容
    article_text = read_article_from_file(file_path)
    
    # 先生成文章总结
    summary = generate_article_summary(article_text)
    
    # 生成文章评估
    assessment = generate_article_assessment(article_text, summary)
    
    # 打印评估结果
    print("文章价值评估:")
    print("-" * 50)
    print(assessment)
    
    # 提取评估分数
    scores = extract_assessment_scores(assessment)
    if scores:
        print("\n评估分数:")
        print("-" * 50)
        print(f"创新性得分: {scores.innovation_score}")
        print(f"实用性得分: {scores.practicality_score}")
        print(f"完整性得分: {scores.completeness_score}")
        print(f"可信度得分: {scores.credibility_score}")
        print(f"特定需求得分: {scores.specific_needs_score}")
        print(f"最终总分: {scores.total_score}")