from vector_db_utils import get_vector_db_client, load_summaries

def ingest_documents(batch_size: int = 1000):
    """将文档导入到向量数据库中"""
    client, custom_ef = get_vector_db_client(batch_size)
    
    # 加载摘要文件
    summaries_dir = "./summaries"
    documents, file_names = load_summaries(summaries_dir)

    # 获取或创建集合
    collection_name = "articles_collection"
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