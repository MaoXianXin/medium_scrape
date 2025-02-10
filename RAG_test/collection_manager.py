from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class CollectionManager:
    """管理向量数据库集合的工具类"""
    
    def __init__(self, openai_api_key, base_url=None):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key,
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        
        self.vector_store_children = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db_children",
            collection_name="child_chunks"
        )
        
        self.vector_store_parents = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db_parents",
            collection_name="parent_chunks"
        )

    def get_collection_stats(self) -> dict:
        """获取向量数据库集合的统计信息
        
        Returns:
            dict: 包含以下统计信息:
                - total_documents: 文档总数(父块和子块)
                - parent_chunks: 父块数量
                - child_chunks: 子块数量
                - unique_sources: 不同源文件数量及清单
        """
        # 获取父块和子块的所有文档
        parent_results = self.vector_store_parents.get()
        child_results = self.vector_store_children.get()
        
        # 统计文档数
        parent_docs = len(parent_results["ids"]) if "ids" in parent_results else 0
        child_docs = len(child_results["ids"]) if "ids" in child_results else 0
        
        # 统计不同源文件
        unique_sources = set()
        for results in [parent_results, child_results]:
            if "metadatas" in results:
                for metadata in results["metadatas"]:
                    if metadata and "source" in metadata:
                        unique_sources.add(metadata["source"])
        
        return {
            "total_documents": parent_docs + child_docs,
            "parent_chunks": parent_docs,
            "child_chunks": child_docs,
            "unique_sources": {
                "count": len(unique_sources),
                "sources": list(unique_sources)
            }
        }
    
    def reset_collections(self):
        """清空向量数据库集合并重置状态"""
        # 重置向量存储
        self.vector_store_children.reset_collection()
        self.vector_store_parents.reset_collection()


# 初始化 CollectionManager
manager = CollectionManager(
    openai_api_key="sk-noerGmiAt3J8SQdnj1UI74K4ixZhB55OUuEp6rfa85BOjVcI",
    base_url="https://zzzzapi.com/v1"
)

# 获取集合统计信息
stats = manager.get_collection_stats()
print("文档总数:", stats["total_documents"])
print("父块数量:", stats["parent_chunks"])
print("子块数量:", stats["child_chunks"])
print("源文件数量:", stats["unique_sources"]["count"])
print("源文件列表:", stats["unique_sources"]["sources"])

# 重置集合
manager.reset_collections()