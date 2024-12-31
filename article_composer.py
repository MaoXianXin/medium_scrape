from vector_db.article_search import search_similar_articles
from utils import OpenAIClient
import os

class ArticleComposer:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.gptapi.us/v1",
        model_name: str = "gpt-4o-mini",
        articles_dir: str = "./articles",
        summaries_dir: str = "./summaries",
        vector_db_path: str = "./vector_db/chroma_db",
        collection_name: str = "articles_collection",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.openai_client = OpenAIClient(api_key, base_url, model_name)
        self.articles_dir = articles_dir
        self.summaries_dir = summaries_dir
        self.api_key = api_key
        self.base_url = base_url
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
    def _read_article(self, article_name: str) -> str:
        """读取文章内容"""
        file_path = os.path.join(self.articles_dir, f"{article_name}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    def _read_framework(self, article_name: str) -> str:
        """读取预生成的知识框架"""
        file_path = os.path.join(self.summaries_dir, f"{article_name}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _save_article(self, topic: str, content: str) -> str:
        """保存生成的文章
        
        Args:
            topic: 文章主题
            content: 文章内容
        Returns:
            保存的文件路径
        """
        # 将主题中的特殊字符替换为下划线
        safe_topic = "".join(c if c.isalnum() else '_' for c in topic)
        file_path = f"generated_{safe_topic}.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return file_path
    
    def compose_article(self, topic: str, batch_size: int = 1000) -> str:
        """基于主题生成新文章
        
        Args:
            topic: 文章主题
            batch_size: 向量搜索的批处理大小
        Returns:
            生成的文章内容
        """
        # 1. 搜索相关文章
        similar_articles = search_similar_articles(
            query=topic,
            top_k=3,
            batch_size=batch_size,
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.embedding_model,
            db_path=self.vector_db_path,
            collection_name=self.collection_name
        )
        
        # 2. 读取文章内容和对应的知识框架
        articles_data = []
        for article_name, similarity in similar_articles:
            content = self._read_article(article_name)
            framework = self._read_framework(article_name)
            articles_data.append({
                "name": article_name,
                "content": content,
                "framework": framework,
                "similarity": similarity
            })
            
        # 3. 基于已有框架和原文生成新文章
        compose_prompt = """你是一个专业的技术文章作者。请基于给定的多篇文章及其知识框架，
创作一篇新的文章。要求：
1. 参考相似度较高的文章内容和框架作为主要结构
2. 合理融合其他文章的补充观点和内容
3. 保持逻辑清晰，层次分明
4. 使用专业且准确的术语
5. 确保内容的完整性和连贯性"""
        
        compose_messages = [
            {"role": "user", "content": f"""
主题: {topic}

参考文章及其知识框架:
{"".join([f'''
文章 {i+1} (相似度: {data["similarity"]:.4f}):
知识框架:
{data["framework"]}

原文内容:
{data["content"]}
---
''' for i, data in enumerate(articles_data)])}

请基于以上内容创作新文章。
"""}
        ]
        
        new_article = self.openai_client.get_completion(
            compose_messages,
            system_prompt=compose_prompt
        ).content
        
        # 保存生成的文章
        saved_path = self._save_article(topic, new_article)
        print(f"文章已保存至: {saved_path}")

        return new_article


if __name__ == "__main__":
    # 使用示例
    composer = ArticleComposer(
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        vector_db_path="./vector_db/chroma_db",
        collection_name="articles_collection",
        model_name="gpt-4o-mini"
    )

    new_article = composer.compose_article("注意力机制种类")
    print(new_article)