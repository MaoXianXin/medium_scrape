import hashlib
import pymysql

class ArticleDBManager:
    """
    文章数据库管理器，负责将文章分析结果保存到数据库
    """
    
    def __init__(self, db_config=None):
        """
        初始化数据库管理器
        
        参数:
            db_config: 数据库配置字典，如果为None则使用默认配置
        """
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
    
    def save_analysis_result(self, article_text, result):
        """
        将分析结果保存到MySQL数据库
        
        参数:
            article_text: 文章原文
            result: 分析结果字典
        
        返回:
            成功返回True，失败返回False
        """
        # 计算文章内容的哈希值作为唯一标识
        content_hash = hashlib.sha256(article_text.encode('utf-8')).hexdigest()
        
        # 从文件路径中提取文章标题（简单处理，实际可能需要更复杂的逻辑）
        article_path = result['article_path']
        article_title = article_path.split('/')[-1].replace('.txt', '')
        
        # 准备标签数据
        tags = result['tags']
        technical_tags = ','.join(tags.get('技术标签', []))
        topic_tags = ','.join(tags.get('主题标签', []))
        application_tags = ','.join(tags.get('应用标签', []))
        
        # 准备分数数据
        scores = result['scores']
        
        try:
            # 连接到MySQL数据库
            connection = pymysql.connect(**self.db_config)
            
            with connection.cursor() as cursor:
                # 检查是否已存在相同内容哈希的记录
                check_sql = f"SELECT id FROM {self.table_name} WHERE article_content_hash = %s"
                cursor.execute(check_sql, (content_hash,))
                existing_record = cursor.fetchone()
                
                if existing_record:
                    # 更新现有记录
                    update_sql = f"""
                    UPDATE {self.table_name} SET
                        article_title = %s,
                        article_path = %s,
                        summary = %s,
                        assessment = %s,
                        innovation_score = %s,
                        practicality_score = %s,
                        completeness_score = %s,
                        credibility_score = %s,
                        specific_needs_score = %s,
                        total_score = %s,
                        technical_tags = %s,
                        topic_tags = %s,
                        application_tags = %s
                    WHERE article_content_hash = %s
                    """
                    cursor.execute(update_sql, (
                        article_title,
                        article_path,
                        result['summary'],
                        result['assessment'],
                        scores['innovation_score'],
                        scores['practicality_score'],
                        scores['completeness_score'],
                        scores['credibility_score'],
                        scores['specific_needs_score'],
                        scores['total_score'],
                        technical_tags,
                        topic_tags,
                        application_tags,
                        content_hash
                    ))
                else:
                    # 插入新记录
                    insert_sql = f"""
                    INSERT INTO {self.table_name} (
                        article_title, article_content_hash, article_path, article_content,
                        summary, assessment, innovation_score, practicality_score,
                        completeness_score, credibility_score, specific_needs_score,
                        total_score, technical_tags, topic_tags, application_tags
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_sql, (
                        article_title,
                        content_hash,
                        article_path,
                        article_text,
                        result['summary'],
                        result['assessment'],
                        scores['innovation_score'],
                        scores['practicality_score'],
                        scores['completeness_score'],
                        scores['credibility_score'],
                        scores['specific_needs_score'],
                        scores['total_score'],
                        technical_tags,
                        topic_tags,
                        application_tags
                    ))
                
                # 提交事务
                connection.commit()
                print(f"成功保存分析结果到数据库，文章哈希: {content_hash}")
                return True
                
        except Exception as e:
            print(f"数据库操作失败: {e}")
            return False
        finally:
            if 'connection' in locals() and connection.open:
                connection.close() 

    def check_article_exists(self, article_text):
        """
        检查文章是否已存在于数据库中
        
        参数:
            article_text: 文章原文
            
        返回:
            如果文章存在返回记录ID，否则返回None
        """
        # 计算文章内容的哈希值
        content_hash = hashlib.sha256(article_text.encode('utf-8')).hexdigest()
        
        try:
            # 连接到MySQL数据库
            connection = pymysql.connect(**self.db_config)
            
            with connection.cursor() as cursor:
                # 检查是否已存在相同内容哈希的记录
                check_sql = f"SELECT id FROM {self.table_name} WHERE article_content_hash = %s"
                cursor.execute(check_sql, (content_hash,))
                existing_record = cursor.fetchone()
                
                if existing_record:
                    return existing_record['id']
                return None
                
        except Exception as e:
            print(f"检查文章存在性时发生错误: {e}")
            return None
        finally:
            if 'connection' in locals() and connection.open:
                connection.close() 