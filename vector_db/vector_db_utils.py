import os
import chromadb
from vector_utils import CustomOpenAIEmbeddingFunction

def get_vector_db_client(
    batch_size: int = 1000,
    api_key: str = None,
    base_url: str = "https://www.gptapi.us/v1",
    model_name: str = "text-embedding-3-small",
    db_path: str = "./chroma_db"
):
    """初始化并返回ChromaDB客户端和embedding函数
    
    Args:
        batch_size: 批处理大小
        api_key: OpenAI API密钥
        base_url: API基础URL
        model_name: 使用的模型名称
        db_path: ChromaDB存储路径
    """
    client = chromadb.PersistentClient(path=db_path)
    custom_ef = CustomOpenAIEmbeddingFunction(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        batch_size=batch_size
    )
    return client, custom_ef

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