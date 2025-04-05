import pymysql
from keyword_matching_engine import calculate_match_score

class ArticleSearchEngine:
    def __init__(self, db_config=None):
        # 默认数据库配置
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 3306,
            'user': 'phpmyadmin',
            'password': '123',
            'database': 'phpmyadmin',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
            'table_name': 'article_analysis_results'
        }
        # 从配置中提取表名
        self.table_name = self.db_config.pop('table_name', 'article_analysis_results')
        
    def search_articles_by_keyword(self, keyword, limit=10):
        """
        根据关键词搜索文章，并按匹配分数排序返回文章标题
        
        参数:
        keyword: 用户输入的关键词
        limit: 返回结果的最大数量，默认为10
        
        返回:
        list: 包含(文章标题, 匹配分数)元组的列表，按分数降序排序
        """
        try:
            # 连接数据库
            connection = pymysql.connect(**self.db_config)
            
            with connection.cursor() as cursor:
                # 查询所有文章的标题和标签信息
                query = f"""
                SELECT id, article_title, technical_tags, topic_tags, application_tags 
                FROM {self.table_name}
                """
                cursor.execute(query)
                articles = cursor.fetchall()
                
                # 计算每篇文章的匹配分数
                scored_articles = []
                for article in articles:
                    score = calculate_match_score(keyword, article)
                    if score > 0:  # 只返回有匹配度的文章
                        scored_articles.append((article['article_title'], score, article['id']))
                
                # 按匹配分数降序排序
                scored_articles.sort(key=lambda x: x[1], reverse=True)
                
                # 限制返回数量
                return scored_articles[:limit]
                
        except Exception as e:
            print(f"搜索文章时出错: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_article_details(self, article_id):
        """
        获取指定ID文章的详细信息
        
        参数:
        article_id: 文章ID
        
        返回:
        dict: 文章的详细信息
        """
        try:
            connection = pymysql.connect(**self.db_config)
            
            with connection.cursor() as cursor:
                query = f"""
                SELECT * FROM {self.table_name}
                WHERE id = %s
                """
                cursor.execute(query, (article_id,))
                return cursor.fetchone()
                
        except Exception as e:
            print(f"获取文章详情时出错: {e}")
            return None
        finally:
            if connection:
                connection.close()


# 使用示例
if __name__ == "__main__":
    search_engine = ArticleSearchEngine()
    
    # 测试关键词搜索
    keyword = "知识图谱"
    results = search_engine.search_articles_by_keyword(keyword, limit=5)
    
    print(f"关键词 '{keyword}' 的搜索结果:")
    for i, (title, score, article_id) in enumerate(results, 1):
        print(f"{i}. 标题: {title} (匹配分数: {score:.4f})")
        
        # 获取并显示文章详情
        article = search_engine.get_article_details(article_id)
        if article:
            print(f"   技术标签: {article['technical_tags']}")
            print(f"   主题标签: {article['topic_tags']}")
            print(f"   应用标签: {article['application_tags']}")
            print(f"   总评分: {article['total_score']}")
            print()