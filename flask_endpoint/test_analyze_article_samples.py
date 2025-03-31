import re
from dialog_module.utils import read_article_from_file, create_custom_llm
from test_article_summary_samples import generate_article_summary, extract_tags_from_summary
from test_article_assessment_samples import generate_article_assessment
from score_extractor import extract_assessment_scores
from article_db_manager import ArticleDBManager

class ArticleAnalyzer:
    """
    文章分析器类，用于收集文章的价值评分和标签
    """
    
    def __init__(self):
        """初始化文章分析器"""
        # 使用自定义模型服务
        self.custom_llm = create_custom_llm()
        # 初始化数据库管理器
        self.db_manager = ArticleDBManager()
    
    def analyze_article(self, article_path, force_reanalyze=False):
        """
        分析文章，收集价值评分和标签
        
        参数:
            article_path: 文章文件路径
            force_reanalyze: 是否强制重新分析，即使文章已存在
            
        返回:
            包含文章分析结果的字典，如果文章已存在且不强制重新分析则返回None
        """
        # 读取文章内容
        article_text = read_article_from_file(article_path)
        
        # 检查文章是否已存在于数据库中
        if not force_reanalyze:
            article_id = self.db_manager.check_article_exists(article_text)
            if article_id:
                print(f"文章已存在于数据库中(ID: {article_id})，跳过分析过程。")
                return None
        
        # 生成文章总结
        summary = generate_article_summary(article_text, llm=self.custom_llm)
        # 过滤掉<think>...</think>内容
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        
        # 从总结中提取标签
        tags = extract_tags_from_summary(summary, llm=self.custom_llm)
        
        # 构建标签字典
        if isinstance(tags, dict) or hasattr(tags, "__dict__"):
            tags_dict = tags
        else:
            # 如果是字符串（解析失败），返回空字典
            tags_dict = {}
        
        # 生成文章评估
        assessment = generate_article_assessment(article_text, summary, llm=self.custom_llm)
        # 过滤掉<think>...</think>内容
        assessment = re.sub(r'<think>.*?</think>', '', assessment, flags=re.DOTALL)
        
        # 提取评估分数
        scores = extract_assessment_scores(assessment, llm=self.custom_llm)
        
        # 构建分析结果
        result = {
            "article_path": article_path,
            "summary": summary,
            "tags": tags_dict,
            "assessment": assessment,
            "scores": {
                "innovation_score": scores.innovation_score if scores else None,
                "practicality_score": scores.practicality_score if scores else None,
                "completeness_score": scores.completeness_score if scores else None,
                "credibility_score": scores.credibility_score if scores else None,
                "specific_needs_score": scores.specific_needs_score if scores else None,
                "total_score": scores.total_score if scores else None
            }
        }
        
        # 保存结果到数据库
        self.db_manager.save_analysis_result(article_text, result)
        
        return result

# 示例使用
if __name__ == "__main__":
    # 创建文章分析器实例
    analyzer = ArticleAnalyzer()
    
    # 单篇文章分析示例
    article_path = "/home/mao/workspace/medium_scrape/articles/5-extremely-useful-plots-for-data-scientists-that-you-never-knew-existed-5b92498a878f.txt"
    
    # 可以通过参数控制是否强制重新分析
    force_reanalyze = False  # 设置为True将强制重新分析
    result = analyzer.analyze_article(article_path, force_reanalyze)
    
    # 如果文章已存在且不强制重新分析，result将为None
    if result is None:
        print("文章已存在，未进行重新分析。")
    else:
        # 打印分析结果
        print("\n文章分析结果:")
        print("-" * 50)
        print(f"文章路径: {result['article_path']}")
        print(f"创新性得分: {result['scores']['innovation_score']}")
        print(f"实用性得分: {result['scores']['practicality_score']}")
        print(f"完整性得分: {result['scores']['completeness_score']}")
        print(f"可信度得分: {result['scores']['credibility_score']}")
        print(f"特定需求得分: {result['scores']['specific_needs_score']}")
        print(f"最终总分: {result['scores']['total_score']}")
        print("\n技术标签:", ", ".join(result['tags'].get('技术标签', [])))
        print("主题标签:", ", ".join(result['tags'].get('主题标签', [])))
        print("应用标签:", ", ".join(result['tags'].get('应用标签', [])))