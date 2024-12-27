from vector_db_utils import get_vector_db_client, load_summaries

def ingest_documents(
    batch_size: int = 1000,
    summaries_dir: str = "./summaries",
    collection_name: str = "articles_collection",
    api_key: str = None,
    base_url: str = "https://www.gptapi.us/v1",
    model_name: str = "text-embedding-3-small",
    db_path: str = "./chroma_db"
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
    """
    client, custom_ef = get_vector_db_client(
        batch_size=batch_size,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        db_path=db_path
    )
    
    # 加载摘要文件
    documents, file_names = load_summaries(summaries_dir)

    # 获取或创建集合
    try:
        collection = client.get_collection(collection_name, embedding_function=custom_ef)
        # 获取现有文档
        existing_docs = collection.get()
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
        
        for i, (doc, fname) in enumerate(zip(documents, file_names)):
            if fname not in existing_files:
                new_docs.append(doc)
                new_ids.append(f"doc_{len(existing_files) + i}")
                new_metadatas.append({"file_name": fname})
            elif existing_files[fname][0] != doc:
                update_docs.append(doc)
                update_ids.append(existing_files[fname][1])
                update_metadatas.append({"file_name": fname})
        
        # 添加新文档
        if new_docs:
            collection.add(
                documents=new_docs,
                ids=new_ids,
                metadatas=new_metadatas
            )
        
        # 更新已有文档
        if update_docs:
            collection.update(
                documents=update_docs,
                ids=update_ids,
                metadatas=update_metadatas
            )
            
    except Exception as e:
        # 如果集合不存在，创建新的集合
        collection = client.create_collection(collection_name, embedding_function=custom_ef)
        collection.add(
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))],
            metadatas=[{"file_name": fname} for fname in file_names]
        )

if __name__ == "__main__":
    ingest_documents(batch_size=10)