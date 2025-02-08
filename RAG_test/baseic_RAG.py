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
from langchain_core.vectorstores import InMemoryVectorStore
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

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
        self.llm = ChatOpenAI(
            model="claude-3-5-sonnet",
            temperature=0,
            openai_api_key=openai_api_key,
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        
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
    
    # 用户查询
    query = "What is LangSmith?"
    print("\n用户查询:", query)
    
    # 相似度搜索获取相关文档
    similar_docs = processor.similarity_search(query)
    
    # 将检索到的文档合并为文本
    docs_text = "\n".join(doc.page_content for doc in similar_docs)
    
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
        print(doc.page_content[:200] + "...")
    
    print("\nAI 回答:")
    print(response.content)

if __name__ == "__main__":
    main()