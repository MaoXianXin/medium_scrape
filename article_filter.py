from utils import OpenAIClient, ArticleFilter

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
    search_topic = "深度学习中的注意力机制和MoE架构"  # 搜索主题
    threshold = 0.6  # 相关度阈值
    
    # 可选的系统提示词
    system_prompt = """你是一个专业的文章相关度分析助手。
    请基于以下几点来评估相关度：
    1. 主题的直接相关性
    2. 核心概念的重叠程度
    3. 技术领域的匹配度
    请返回一个0到1之间的分数。"""
    
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