"""
文档的处理流程为: 文档加载->文档分割->嵌入->向量存储

graph TD
    A[开始] --> B[初始化 DocumentPipeline]
    B --> C{配置初始化}
    C --> |1| D[设置 OpenAI API]
    C --> |2| E[初始化 Embeddings 模型]
    C --> |3| F[初始化文本分割器]
    C --> |4| G[初始化向量存储]
    C --> |5| G2[初始化存储容器<br>parent_chunks<br>parent_to_children]
    
    H[process_document] --> I[加载文档<br>load_document]
    I --> J[PDF加载器处理]
    J --> K[添加源文件信息]
    
    K --> L[分割文档<br>split_documents]
    L --> M[创建父块]
    M --> N[为每个父块创建子块]
    N --> O[生成内容哈希值]
    O --> P[建立父子块关系]
    
    P --> P2[存储处理结果]
    P2 --> |存储父块| P3[更新 parent_chunks]
    P2 --> |存储映射关系| P4[更新 parent_to_children]
    P2 --> |返回所有子块| P5[生成 splits]
    
    P5 --> Q[向量存储处理]
    Q --> R[添加文档到向量存储]
    
    R --> S[返回处理结果]
    S --> S1[返回状态: success]
    S --> S2[返回父块数量]
    S --> S3[返回子块数量]
    S --> T[结束]
    
    subgraph 文本分割配置
    F1[父块分割器<br>chunk_size=1000<br>overlap=200]
    F2[子块分割器<br>chunk_size=100<br>overlap=20]
    end
    
    subgraph 元数据处理
    O1[父块ID: parent_hash]
    O2[子块ID: child_hash]
    O3[关联父子块ID]
    end
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import hashlib

class DocumentPipeline:
    def __init__(self, openai_api_key, base_url=None):
        # 初始化配置
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        
        # 初始化分割器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            add_start_index=True
        )
        
        # 初始化向量存储
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",  # 指定存储目录
            collection_name="my_collection"    # 可选：指定集合名称
        )
        
        # 存储文档块的引用
        self.parent_chunks = []
        self.parent_to_children = {}

    def load_document(self, file_path):
        """加载PDF文档"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 添加源文件信息
        for doc in docs:
            doc.metadata['source'] = file_path
            doc.metadata['page'] = doc.metadata.get('page', 0)
        return docs

    @staticmethod
    def get_content_hash(text):
        """生成文本内容的哈希值"""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def split_documents(self, documents):
        """将文档分割为父块和子块"""
        # 创建父块
        parent_chunks = self.parent_splitter.split_documents(documents)
        
        # 为每个父块创建子块
        all_children = []
        parent_to_children = {}
        
        for parent in parent_chunks:
            parent_hash = self.get_content_hash(parent.page_content)
            parent.metadata['chunk_id'] = f'parent_{parent_hash}'
            
            children = self.child_splitter.split_documents([parent])
            
            for child in children:
                child_hash = self.get_content_hash(child.page_content)
                child.metadata['parent_id'] = parent.metadata['chunk_id']
                child.metadata['chunk_id'] = f'child_{child_hash}'
                all_children.append(child)
            
            parent_to_children[parent.metadata['chunk_id']] = children
        
        self.parent_chunks = parent_chunks
        self.parent_to_children = parent_to_children
        
        return all_children

    def process_document(self, file_path):
        """执行完整的文档处理流程"""
        # 1. 加载文档
        docs = self.load_document(file_path)
        
        # 2. 分割文档
        splits = self.split_documents(docs)
        
        # 3. 添加到向量存储
        self.vector_store.add_documents(documents=splits, ids=[doc.metadata['chunk_id'] for doc in splits])
        
        return {
            "status": "success",
            "parent_chunks_count": len(self.parent_chunks),
            "child_chunks_count": len(splits)
        }

    def get_collection_stats(self) -> dict:
        """获取向量数据库集合的统计信息
        
        Returns:
            dict: 包含以下统计信息:
                - total_documents: 文档总数
                - unique_sources: 不同源文件数量及清单
        """
        # 获取所有文档
        results = self.vector_store.get()
        
        # 统计总文档数
        total_docs = len(results["ids"]) if "ids" in results else 0
        
        # 统计不同源文件
        unique_sources = set()
        if "metadatas" in results:
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    unique_sources.add(metadata["source"])
        
        return {
            "total_documents": total_docs,
            "unique_sources": {
                "count": len(unique_sources),
                "sources": list(unique_sources)
            }
        }
    
    def reset_collection(self):
        """清空向量数据库集合并重置状态"""
        # 重置向量存储
        self.vector_store.reset_collection()
        
        # 重置内部状态
        self.parent_chunks = []
        self.parent_to_children = {}

# 使用示例
def main():
    pipeline = DocumentPipeline(
        openai_api_key="sk-noerGmiAt3J8SQdnj1UI74K4ixZhB55OUuEp6rfa85BOjVcI",
        base_url="https://zzzzapi.com/v1"  # 可选
    )
    
    # 获取统计信息
    stats = pipeline.get_collection_stats()
    print(f"文档总数: {stats['total_documents']}")
    print(f"源文件数量: {stats['unique_sources']['count']}")
    print(f"源文件列表: {stats['unique_sources']['sources']}")

    # 重置集合
    # pipeline.reset_collection()

    result = pipeline.process_document("/home/mao/Downloads/Introduction _ 🦜️🔗 LangChain.pdf")
    print(f"文档处理完成：{result}")

if __name__ == "__main__":
    main()