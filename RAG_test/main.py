from document_processor import DocumentProcessor
from retrieval_qa import RetrievalQA

def main():
    api_key = "sk-noerGmiAt3J8SQdnj1UI74K4ixZhB55OUuEp6rfa85BOjVcI"
    base_url = "https://zzzzapi.com/v1"
    
    # 文档处理
    processor = DocumentProcessor(api_key, base_url)
    file_path = "/home/mao/Downloads/LangChain.pdf"
    print("开始处理文档...")
    result = processor.process_document(file_path)
    print(result)
    
    # 检索问答
    qa = RetrievalQA(api_key, base_url)
    query_text = "What is LangSmith?"
    print("\n用户查询:", query_text)
    result = qa.query(query_text)
    
    print("\n检索到的相关文档:")
    for i, doc in enumerate(result['sources'], 1):
        print(f"\n文档 {i}:")
        print(f"Source: {doc['source_info']['source']}, Page: {doc['source_info']['page']}, Chunk ID: {doc['source_info']['chunk_id']}")
        print(doc['document'].page_content)
    
    print("\nAI 回答:")
    print(result['answer'])

if __name__ == "__main__":
    main()