import chromadb
from chromadb.api.types import Documents, EmbeddingFunction
from openai import OpenAI
from typing import List
import os

class CustomOpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 1000,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.batch_size = batch_size

    def __call__(self, texts: Documents) -> List[List[float]]:
        # 确保输入是字符串列表
        if not isinstance(texts, list):
            texts = [texts]
        
        # 处理空输入
        if not texts:
            return []

        # 批量处理文本
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                raise e

        return all_embeddings

def load_summaries(summaries_dir: str) -> tuple[list[str], list[str]]:
    """加载所有摘要文件的内容和对应的文件名"""
    documents = []
    file_names = []
    for file_name in os.listdir(summaries_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(summaries_dir, file_name), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                documents.append(content)
                file_names.append(file_name)
    return documents, file_names

def search_similar_articles(query: str, top_k: int = 3) -> List[tuple[str, float]]:
    """根据查询文本搜索相似的文章"""
    # 初始化 Chroma 客户端和 embedding 函数
    client = chromadb.PersistentClient(path="./chroma_db")
    custom_ef = CustomOpenAIEmbeddingFunction(
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        model_name="text-embedding-3-small"
    )

    # 加载摘要文件
    summaries_dir = "./summaries"
    documents, file_names = load_summaries(summaries_dir)

    # 获取集合
    collection_name = "articles_collection"
    # 如果集合已存在，先删除它
    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(collection_name)
    
    # 创建新的集合
    collection = client.create_collection(collection_name, embedding_function=custom_ef)

    # 添加新文档
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"file_name": fname} for fname in file_names]
    )

    # 查询文档
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    
    # 返回相关文章的文件名和相似度
    similar_articles = [
        (metadata["file_name"].replace('.txt', ''), distance)  # 返回(文件名, 距离)元组
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
    ]
    
    return similar_articles

# 使用示例
if __name__ == "__main__":
    query = "注意力机制"
    similar_articles = search_similar_articles(query, top_k=3)
    print("\n相关文章及其相似度：")
    for article, distance in similar_articles:
        print(f"距离: {distance:.4f} - {article}")