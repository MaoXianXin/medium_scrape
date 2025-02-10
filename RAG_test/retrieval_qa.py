from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
class RetrievalQA:
    def __init__(self, openai_api_key, base_url=None):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url if base_url else "https://api.openai.com/v1"
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
        
        self.llm = ChatOpenAI(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )

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

    def query(self, query_text, search_type="similarity", **search_params):
        """统一的查询接口"""
        if search_type == "similarity":
            similar_docs = self.similarity_search(query_text, **search_params)
        elif search_type == "mmr":
            similar_docs = self.mmr_search(query_text, **search_params)
        elif search_type == "threshold":
            similar_docs = self.similarity_score_threshold_search(query_text, **search_params)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        docs_text = "\n".join(doc['document'].page_content for doc in similar_docs)
        
        system_prompt = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer clear and concise while providing complete information.
        Context: {context}"""
        
        messages = [
            SystemMessage(content=system_prompt.format(context=docs_text)),
            HumanMessage(content=query_text)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            'answer': response.content,
            'sources': similar_docs
        }