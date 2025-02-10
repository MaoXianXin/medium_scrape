from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import hashlib

class DocumentProcessor:
    def __init__(self, openai_api_key, base_url=None):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        
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

    def process_document(self, file_path):
        """完整的文档处理流程"""
        docs = self.load_document(file_path)
        splits = self.split_documents(docs)
        self.add_to_vectorstore(splits)
        return "Document processed successfully" 