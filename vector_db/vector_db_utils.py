import os
import chromadb
from vector_utils import CustomOpenAIEmbeddingFunction

def get_vector_db_client(batch_size: int = 1000):
    """初始化并返回ChromaDB客户端和embedding函数"""
    client = chromadb.PersistentClient(path="./chroma_db")
    custom_ef = CustomOpenAIEmbeddingFunction(
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        model_name="text-embedding-3-small",
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