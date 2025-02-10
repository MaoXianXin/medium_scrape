"""
文档的处理流程为: 文档加载->文档分割->嵌入->向量存储->检索->问答

graph TD
    A[文档处理开始] --> B[初始化 DocumentProcessor]
    
    subgraph 初始化阶段
    B --> C[设置 OpenAI API Key]
    B --> D[初始化组件]
    D -->|1| E[OpenAIEmbeddings<br>model: text-embedding-3-large<br>base_url: 可选]
    D -->|2| F[RecursiveCharacterTextSplitter<br>chunk_size: 1000<br>chunk_overlap: 200<br>add_start_index: true]
    D -->|3| G[InMemoryVectorStore]
    D -->|4| Q[ChatOpenAI<br>model: claude-3-5-sonnet<br>temperature: 0<br>base_url: 可选]
    E --> G
    end
    
    subgraph 文档处理流程
    H[process_document方法] --> I[加载文档<br>load_document]
    I --> J[文档分割<br>split_documents]
    J --> K[添加到向量存储<br>add_to_vectorstore]
    end
    
    subgraph 搜索功能
    L[搜索方法] --> M[相似度搜索<br>similarity_search<br>参数: k=4]
    L --> N[MMR搜索<br>mmr_search<br>参数: k=4, fetch_k=20]
    L --> O[相似度阈值搜索<br>similarity_score_threshold_search<br>参数: score_threshold=0.8]
    M --> P[返回相关文档]
    N --> P
    O --> P
    end

    subgraph 问答功能
    R[问答流程] --> S[用户查询]
    S --> M
    P --> T[合并文档内容]
    T --> U[构建系统提示词]
    U --> V[生成回答<br>ChatOpenAI.invoke]
    Q --> V
    end

    %% 添加子图之间的关系
    F --> J
    G --> K
    
    E -.->|查询向量化| M
    E -.->|查询向量化| N
    E -.->|查询向量化| O
    G -.->|向量存储检索| M
    G -.->|向量存储检索| N
    G -.->|向量存储检索| O

    %% 添加流程顺序
    初始化阶段 --> 文档处理流程
    文档处理流程 --> 搜索功能
    搜索功能 --> 问答功能

    %% 添加说明
    style 初始化阶段 fill:#f9f,stroke:#333,stroke-width:2px
    style 文档处理流程 fill:#bbf,stroke:#333,stroke-width:2px
    style 搜索功能 fill:#bfb,stroke:#333,stroke-width:2px
    style 问答功能 fill:#fbb,stroke:#333,stroke-width:2px
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain.schema import Document
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import hashlib
class DocumentProcessor:
    def __init__(self, openai_api_key, base_url=None):
        # 初始化配置
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 初始化各个组件
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        # 添加父块分割器
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
        # 添加两个 Chroma 集合，分别用于存储父块和子块
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
        
        self.llm = ChatOpenAI(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            openai_api_key=openai_api_key,
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        
        # 初始化 parent_chunks
        self.parent_chunks = []
        
    def load_document(self, file_path):
        """加载文档"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 添加源文件信息
        for doc in docs:
            doc.metadata['source'] = file_path
            doc.metadata['page'] = doc.metadata.get('page', 0)
        return docs
        
    @staticmethod
    def get_content_hash(text):
        """根据文本内容生成哈希值"""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def split_documents(self, documents):
        """分割文档为父块和子块"""
        # 首先创建父块
        parent_chunks = self.parent_splitter.split_documents(documents)
        
        # 为每个父块创建子块
        all_children = []
        
        for parent_idx, parent in enumerate(parent_chunks):
            # 使用内容哈希作为父块ID
            parent_hash = self.get_content_hash(parent.page_content)
            parent.metadata['chunk_id'] = f'parent_{parent_hash}_{parent_idx}'
            
            # 从父块创建子块
            children = self.child_splitter.split_documents([parent])
            
            # 为子块添加父块引用
            for child_idx, child in enumerate(children):
                child_hash = self.get_content_hash(child.page_content)
                child.metadata['parent_id'] = parent.metadata['chunk_id']
                child.metadata['chunk_id'] = f'child_{child_hash}_p{parent_idx}_c{child_idx}'
                all_children.append(child)
        
        # 存储父块
        self.parent_chunks = parent_chunks
        
        return all_children
        
    def add_to_vectorstore(self, documents):
        """将父块和子块分别添加到向量存储"""
        if not hasattr(self, 'parent_chunks') or not self.parent_chunks:
            raise ValueError("Parent chunks not available. Please run split_documents first.")
            
        # 分离父块和子块
        parent_docs = self.parent_chunks
        child_docs = documents
        
        try:
            # 存储父块
            self.vector_store_parents.add_documents(
                documents=parent_docs,
                ids=[doc.metadata['chunk_id'] for doc in parent_docs]
            )
            
            # 存储子块
            self.vector_store_children.add_documents(
                documents=child_docs,
                ids=[doc.metadata['chunk_id'] for doc in child_docs]
            )
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def get_parent_chunk(self, child_doc):
        """根据子块获取对应的父块"""
        parent_id = child_doc.metadata.get('parent_id')
        if not hasattr(self, 'parent_chunks') or not self.parent_chunks:
            raise ValueError("Parent chunks not available. Please run split_documents first.")
            
        if parent_id:
            for parent in self.parent_chunks:
                if parent.metadata['chunk_id'] == parent_id:
                    return parent
        return child_doc

    def get_formatted_source_info(self, doc):
        """获取格式化的源文件信息"""
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'Unknown page')
        chunk_id = doc.metadata.get('chunk_id', 'Unknown chunk')
        return {
            'source': source,
            'page': page,
            'chunk_id': chunk_id
        }

    def _get_parent_document_info(self, parent_id):
        """Helper method to get parent document info from parent_id"""
        parent_docs = self.vector_store_parents.get(
            ids=[parent_id],
            include=["documents", "metadatas"]
        )
        
        if parent_docs and parent_docs['documents']:
            parent_doc = Document(
                page_content=parent_docs['documents'][0],
                metadata=parent_docs['metadatas'][0]
            )
            return {
                'document': parent_doc,
                'source_info': self.get_formatted_source_info(parent_doc)
            }
        return None

    def similarity_search(self, query, k=4):
        """使用子块进行相似度搜索，返回父块及其源信息"""
        child_results = self.vector_store_children.similarity_search(query, k=k)
        
        seen_parent_ids = set()
        results = []
        
        for doc in child_results:
            parent_id = doc.metadata['parent_id']
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_info = self._get_parent_document_info(parent_id)
                if parent_info:
                    results.append(parent_info)
        
        return results

    def mmr_search(self, query, k=4, fetch_k=20):
        """使用子块进行最大边际相关性搜索，返回父块及其源信息"""
        child_results = self.vector_store_children.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k
        )
        
        seen_parent_ids = set()
        results = []
        
        for doc in child_results:
            parent_id = doc.metadata['parent_id']
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_info = self._get_parent_document_info(parent_id)
                if parent_info:
                    results.append(parent_info)
                    
        return results
        
    def similarity_score_threshold_search(self, query, score_threshold=0.8):
        """使用子块进行相似度阈值搜索，返回父块及其源信息"""
        results = self.vector_store_children.similarity_search_with_score(query)
        seen_parent_ids = set()
        filtered_results = []
        
        for doc, score in results:
            if score >= score_threshold:
                parent_id = doc.metadata['parent_id']
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    parent_info = self._get_parent_document_info(parent_id)
                    if parent_info:
                        parent_info['score'] = score  # 添加相似度分数
                        filtered_results.append(parent_info)
        
        return filtered_results

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
        "sk-noerGmiAt3J8SQdnj1UI74K4ixZhB55OUuEp6rfa85BOjVcI",
        base_url="https://zzzzapi.com/v1"  # 可选参数
    )
    
    # 处理文档
    file_path = "/home/mao/Downloads/LangChain.pdf"
    print("开始处理文档...")
    result = processor.process_document(file_path)
    print(result)
    
    # 用户查询
    query = "What is LangSmith?"
    print("\n用户查询:", query)
    
    # 相似度搜索获取相关文档
    similar_docs = processor.similarity_search(query)
    
    # 将检索到的文档合并为文本
    docs_text = "\n".join(doc['document'].page_content for doc in similar_docs)
    
    # 定义系统提示词
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep the answer clear and concise while providing complete information.
    Context: {context}"""
    
    system_prompt_fmt = system_prompt.format(context=docs_text)
    
    # 生成回答
    messages = [
        SystemMessage(content=system_prompt_fmt),
        HumanMessage(content=query)
    ]
    response = processor.llm.invoke(messages)
    
    print("\n检索到的相关文档:")
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n文档 {i}:")
        print(f"Source: {doc['source_info']['source']}, Page: {doc['source_info']['page']}, Chunk ID: {doc['source_info']['chunk_id']}")
        print(doc['document'].page_content + "...")
    
    print("\nAI 回答:")
    print(response.content)

if __name__ == "__main__":
    main()