from vector_db_utils import get_vector_db_client, load_summaries
import logging
import hashlib

def generate_doc_id(content: str, filename: str) -> str:
    """生成文档的唯一ID
    
    Args:
        content: 文档内容
        filename: 文件名
    
    Returns:
        str: 基于内容和文件名的哈希ID
    """
    combined = f"{filename}:{content}"
    return hashlib.sha256(combined.encode()).hexdigest()

def ingest_documents(
    batch_size: int = 1000,
    summaries_dir: str = "./summaries",
    collection_name: str = "articles_collection",
    api_key: str = None,
    base_url: str = "https://www.gptapi.us/v1",
    model_name: str = "text-embedding-3-small",
    db_path: str = "./chroma_db",
    force_update: bool = False
):
    """将文档导入到向量数据库中
    
    Args:
        batch_size: 批处理大小
        summaries_dir: 摘要文件目录路径
        collection_name: 向量数据库集合名称
        api_key: OpenAI API密钥
        base_url: API基础URL
        model_name: 使用的模型名称
        db_path: ChromaDB存储路径
        force_update: 是否强制更新所有文档，即使内容没有变化
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"开始导入文档到集合: {collection_name}")
    client, custom_ef = get_vector_db_client(
        batch_size=batch_size,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        db_path=db_path
    )
    
    # 加载摘要文件
    documents, file_names = load_summaries(summaries_dir)
    logging.info(f"已加载 {len(documents)} 个文档")

    # 获取或创建集合
    try:
        collection = client.get_collection(collection_name, embedding_function=custom_ef)
        logging.info("成功获取现有集合")
        
        # 获取现有文档
        existing_docs = collection.get()
        logging.info(f"集合中已有 {len(existing_docs['documents'])} 个文档")
        
        existing_metadata = existing_docs["metadatas"]
        existing_files = {meta["file_name"]: (doc, id) 
                         for meta, doc, id in zip(existing_metadata, 
                                                existing_docs["documents"],
                                                existing_docs["ids"])} if existing_metadata else {}
        
        # 处理新增和更新的文档
        new_docs = []
        new_ids = []
        new_metadatas = []
        update_docs = []
        update_ids = []
        update_metadatas = []
        
        for doc, fname in zip(documents, file_names):
            doc_id = generate_doc_id(doc, fname)
            if fname not in existing_files:
                new_docs.append(doc)
                new_ids.append(doc_id)
                new_metadatas.append({"file_name": fname})
            elif force_update or existing_files[fname][0] != doc:
                update_docs.append(doc)
                update_ids.append(doc_id)
                update_metadatas.append({"file_name": fname})
        
        # 添加新文档
        if new_docs:
            logging.info(f"添加 {len(new_docs)} 个新文档")
            collection.add(
                documents=new_docs,
                ids=new_ids,
                metadatas=new_metadatas
            )
        
        # 更新已有文档
        if update_docs:
            logging.info(f"更新 {len(update_docs)} 个现有文档")
            collection.update(
                documents=update_docs,
                ids=update_ids,
                metadatas=update_metadatas
            )
            
    except Exception as e:
        logging.warning(f"集合不存在，创建新集合: {str(e)}")
        collection = client.create_collection(collection_name, embedding_function=custom_ef)
        logging.info(f"添加 {len(documents)} 个文档到新集合")
        collection.add(
            documents=documents,
            ids=[generate_doc_id(doc, fname) for doc, fname in zip(documents, file_names)],
            metadatas=[{"file_name": fname} for fname in file_names]
        )
    
    logging.info("文档导入完成")

if __name__ == "__main__":
    ingest_documents(
        batch_size=10,
        summaries_dir="../summaries",
        collection_name="articles_collection",
        api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4",
        base_url="https://www.gptapi.us/v1",
        model_name="text-embedding-3-small",
        db_path="./chroma_db",
        force_update=False
    )