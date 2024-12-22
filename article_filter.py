from utils import OpenAIClient, ArticleFilter

"""
article_filter.py 是文章筛选器的实现。
它使用OpenAI API来评估文章与给定主题的相关度，并返回筛选后的文章列表。
search_topic的写法
threshold的阈值
system_prompt的提示词
"""

def main():
    # 初始化OpenAI客户端
    ai_client = OpenAIClient(
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        model_name="gpt-4o-mini"
    )
    
    # 创建文章筛选器实例
    article_filter = ArticleFilter(ai_client)
    
    # 定义搜索参数
    summaries_dir = "./summaries"  # 知识框架文件目录
    search_topic = "注意力机制"  # 搜索主题
    threshold = 0.6  # 相关度阈值
    
    # 可选的系统提示词
    system_prompt = """你是一个专业的文献相关度分析专家。你的任务是评估知识框架图与搜索主题的相关度。

评分标准：
1. 主题相关性(40分)：
   - 知识框架的核心主题与搜索主题的直接关联程度
   - 关键词的重叠度和语义相似度

2. 内容深度(30分)：
   - 知识框架中包含的与搜索主题相关的深入讨论
   - 相关概念、理论或方法的完整性

3. 应用价值(30分)：
   - 知识框架中的内容对理解搜索主题的实际帮助程度
   - 是否包含可操作的见解或实践指导

请根据以上标准进行评分，将总分除以100得出最终的相关度分数(0-1之间)。
只返回最终的相关度分数，例如：0.75"""
    
    try:
        # 执行文章筛选
        filtered_articles = article_filter.filter_articles(
            summaries_dir=summaries_dir,
            search_topic=search_topic,
            threshold=threshold,
            system_prompt=system_prompt
        )
        
        # 打印筛选结果
        print(f"\n找到 {len(filtered_articles)} 篇相关文章：\n")
        
        for idx, article in enumerate(filtered_articles, 1):
            print(f"=== 文章 {idx} ===")
            print(f"文件名: {article['filename']}")
            print(f"相关度: {article['relevance_score']:.2f}")
            print(f"知识框架预览: {article['framework'][:200]}...")  # 只显示前200个字符
            print("-" * 50)
            
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()