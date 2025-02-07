"""
文档的处理流程为: 文档加载->文档分割->嵌入->向量存储->检索
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os

class DocumentProcessor:
    def __init__(self, openai_api_key, base_url=None):
        # 初始化配置
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 初始化各个组件
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 每个文本块最大1000个字符
            chunk_overlap=200,
            add_start_index=True
        )
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
    def load_document(self, file_path):
        """加载文档"""
        loader = PyPDFLoader(file_path)
        return loader.load()
        
    def split_documents(self, documents):
        """分割文档"""
        return self.text_splitter.split_documents(documents)
        
    def add_to_vectorstore(self, documents):
        """将文档添加到向量存储"""
        return self.vector_store.add_documents(documents=documents)
        
    def similarity_search(self, query, k=4):
        """相似度搜索"""
        return self.vector_store.similarity_search(query, k=k)
        
    def mmr_search(self, query, k=4, fetch_k=20):
        """最大边际相关性搜索"""
        return self.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k
        )
        
    def similarity_score_threshold_search(self, query, score_threshold=0.8):
        """相似度阈值搜索"""
        results = self.vector_store.similarity_search_with_score(query)
        return [doc for doc, score in results if score >= score_threshold]

    def process_document(self, file_path):
        """完整的文档处理流程"""
        # 1. 加载文档
        docs = self.load_document(file_path)
        
        # 2. 分割文档
        splits = self.split_documents(docs)
        
        # 3. 添加到向量存储
        self.add_to_vectorstore(splits)
        
        return "Document processed successfully"

# 使用示例
def main():
    # 初始化处理器（添加可选的 base_url 参数）
    processor = DocumentProcessor(
        "sk-jsbIUYq1MwwUheWDDeB01c1a7a2246E3956b6b75762c9f21",
        base_url="https://api.gptapi.us/v1"  # 可选参数
    )
    
    # 处理文档
    file_path = "/home/mao/Downloads/LangChain.pdf"
    print("开始处理文档...")
    result = processor.process_document(file_path)
    print(result)
    
    # 执行搜索
    query = "What is this document about?"
    print("\n执行搜索查询:", query)
    
    # 相似度搜索
    print("\n1. 相似度搜索结果:")
    similar_docs = processor.similarity_search(query)
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n文档 {i}:")
        print(doc.page_content[:200] + "...")  # 只打印前200个字符
    
    # MMR搜索
    print("\n2. MMR搜索结果:")
    mmr_docs = processor.mmr_search(query)
    for i, doc in enumerate(mmr_docs, 1):
        print(f"\n文档 {i}:")
        print(doc.page_content[:200] + "...")
    
    # 阈值搜索
    print("\n3. 阈值搜索结果:")
    threshold_docs = processor.similarity_score_threshold_search(query)
    for i, doc in enumerate(threshold_docs, 1):
        print(f"\n文档 {i}:")
        print(doc.page_content[:200] + "...")
    
    return similar_docs, mmr_docs, threshold_docs

if __name__ == "__main__":
    main()