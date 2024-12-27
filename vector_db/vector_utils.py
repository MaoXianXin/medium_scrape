from chromadb.api.types import Documents, EmbeddingFunction
from openai import OpenAI
from typing import List

class CustomOpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 1000,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.batch_size = batch_size

    def __call__(self, texts: Documents) -> List[List[float]]:
        # 确保输入是字符串列表
        if not isinstance(texts, list):
            texts = [texts]
        
        # 处理空输入
        if not texts:
            return []

        # 添加输入验证
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("All elements in texts must be strings")
            
        # 添加空字符串检查
        if any(not text.strip() for text in texts):
            raise ValueError("Empty strings are not allowed")
            
        # 批量处理文本
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                raise e

        return all_embeddings