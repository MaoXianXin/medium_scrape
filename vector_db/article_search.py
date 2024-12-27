from typing import List
from vector_db_utils import get_vector_db_client

def search_similar_articles(
    query: str, 
    top_k: int = 3, 
    batch_size: int = 1000,
    api_key: str = None,
    base_url: str = "https://www.gptapi.us/v1",
    model_name: str = "text-embedding-3-small",
    db_path: str = "./chroma_db",
    collection_name: str = "articles_collection"
) -> List[tuple[str, float]]:
    """根据查询文本搜索相似的文章
    
    Args:
        query: 查询文本
        top_k: 返回最相似的文章数量
        batch_size: 批处理大小
        api_key: OpenAI API密钥
        base_url: API基础URL
        model_name: 使用的模型名称
        db_path: ChromaDB存储路径
        collection_name: 集合名称
    """
    client, custom_ef = get_vector_db_client(
        batch_size=batch_size,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        db_path=db_path
    )
    
    # 获取集合
    collection = client.get_collection(collection_name, embedding_function=custom_ef)
    
    # 查询文档
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    
    # 返回相关文章的文件名和相似度
    similar_articles = [
        (metadata["file_name"].replace('.txt', ''), distance)
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
    ]
    
    return similar_articles

if __name__ == "__main__":
    query = "注意力机制"
    similar_articles = search_similar_articles(
        query, 
        top_k=3, 
        batch_size=10, 
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4", 
        base_url="https://www.gptapi.us/v1", 
        model_name="text-embedding-3-small", 
        db_path="./chroma_db",
        collection_name="articles_collection"
    )
    print("相关文章及其相似度：")
    for article, distance in similar_articles:
        print(f"距离: {distance:.4f} - {article}")