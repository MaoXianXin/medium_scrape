from typing import List
from vector_db_utils import get_vector_db_client

def search_similar_articles(query: str, top_k: int = 3, batch_size: int = 1000) -> List[tuple[str, float]]:
    """根据查询文本搜索相似的文章"""
    client, custom_ef = get_vector_db_client(batch_size)
    
    # 获取集合
    collection = client.get_collection("articles_collection", embedding_function=custom_ef)
    
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
    similar_articles = search_similar_articles(query, top_k=3, batch_size=10)
    print("相关文章及其相似度：")
    for article, distance in similar_articles:
        print(f"距离: {distance:.4f} - {article}") 