import re
from dialog_module.utils import read_article_from_file, create_custom_llm
from test_article_summary_samples import generate_article_summary, extract_tags_from_summary
from test_article_assessment_samples import generate_article_assessment
from score_extractor import extract_assessment_scores
import json

class ArticleAnalyzer:
    """
    文章分析器类，用于收集文章的价值评分和标签
    """
    
    def __init__(self):
        """初始化文章分析器"""
        # 使用自定义模型服务
        self.custom_llm = create_custom_llm()
    
    def analyze_article(self, article_path):
        """
        分析文章，收集价值评分和标签
        
        参数:
            article_path: 文章文件路径
            
        返回:
            包含文章分析结果的字典
        """
        # 读取文章内容
        article_text = read_article_from_file(article_path)
        
        # 生成文章总结
        summary = generate_article_summary(article_text)
        # 过滤掉<think>...</think>内容
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        
        # 从总结中提取标签
        tags = extract_tags_from_summary(summary)
        
        # 构建标签字典
        if isinstance(tags, dict) or hasattr(tags, "__dict__"):
            tags_dict = tags
        else:
            # 如果是字符串（解析失败），返回空字典
            tags_dict = {}
        
        # 生成文章评估
        assessment = generate_article_assessment(article_text, summary)
        # 过滤掉<think>...</think>内容
        assessment = re.sub(r'<think>.*?</think>', '', assessment, flags=re.DOTALL)
        
        # 提取评估分数
        scores = extract_assessment_scores(assessment)
        
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
        
        return result
    
    def analyze_multiple_articles(self, article_paths):
        """
        批量分析多篇文章
        
        参数:
            article_paths: 文章文件路径列表
            
        返回:
            包含所有文章分析结果的列表
        """
        results = []
        for path in article_paths:
            try:
                result = self.analyze_article(path)
                results.append(result)
                print(f"已完成文章分析: {path}")
            except Exception as e:
                print(f"分析文章 {path} 时出错: {str(e)}")
        
        return results
    
    def save_results_to_json(self, results, output_path):
        """
        将分析结果保存为JSON文件
        
        参数:
            results: 分析结果列表或字典
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"分析结果已保存至: {output_path}")


# 示例使用
if __name__ == "__main__":
    # 创建文章分析器实例
    analyzer = ArticleAnalyzer()
    
    # 单篇文章分析示例
    article_path = "/home/mao/workspace/medium_scrape/articles/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"
    result = analyzer.analyze_article(article_path)
    
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
    
    # 保存结果到JSON文件
    analyzer.save_results_to_json(result, "article_analysis_result.json")
    
    # 批量分析示例
    # article_paths = [
    #     "/path/to/article1.txt",
    #     "/path/to/article2.txt",
    #     "/path/to/article3.txt"
    # ]
    # results = analyzer.analyze_multiple_articles(article_paths)
    # analyzer.save_results_to_json(results, "multiple_articles_analysis.json")