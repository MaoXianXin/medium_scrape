from typing import List, Dict, Union
import numpy as np
import faiss
from pathlib import Path
import pickle
from datetime import datetime
import hashlib
from utils import OpenAIClient


class VectorSearcher:
    def __init__(self, ai_client: OpenAIClient):
        self.ai_client = ai_client
        self.index = None
        self.metadata = {}
        self.vectors_file = "vectors.pkl"
        self.metadata_file = "metadata.pkl"
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 文本的向量表示
        """
        response = self.ai_client.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def _get_files_hash(self, summaries_dir: str) -> str:
        """
        计算目录下所有文件的hash值，用于检测文件变化
        """
        hash_md5 = hashlib.md5()
        summaries_path = Path(summaries_dir)
        
        # 获取所有文件名并排序，确保一致性
        file_paths = sorted(str(p) for p in summaries_path.glob("*.txt"))
        
        for file_path in file_paths:
            # 将文件名和修改时间添加到hash
            stat = Path(file_path).stat()
            file_info = f"{file_path}{stat.st_mtime}"
            hash_md5.update(file_info.encode())
            
        return hash_md5.hexdigest()
        
    def build_index(self, summaries_dir: str, force_rebuild: bool = False):
        """
        构建或加载向量索引
        
        Args:
            summaries_dir: 知识框架文件目录
            force_rebuild: 是否强制重建索引
        """
        current_hash = self._get_files_hash(summaries_dir)
        
        # 尝试加载现有索引
        if not force_rebuild and self._try_load_index(current_hash):
            print("成功加载现有索引")
            return
            
        print("构建新索引...")
        embeddings = []
        ids = []
        current_id = 0
        
        summaries_path = Path(summaries_dir)
        for file_path in summaries_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            framework_part = content.split("第二部分：核心概念定义")[0].strip()
            embedding = self._get_embedding(framework_part)
            
            embeddings.append(embedding)
            ids.append(current_id)
            
            self.metadata[current_id] = {
                'filename': file_path.name,
                'summary_path': str(file_path),
                'article_path': str(file_path).replace('summaries', 'articles'),
                'last_updated': datetime.now().isoformat()
            }
            
            current_id += 1
            
        # 创建和保存索引
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(index)
        self.index.add_with_ids(
            np.array(embeddings, dtype=np.float32),
            np.array(ids, dtype=np.int64)
        )
        
        # 保存索引和metadata
        self._save_index(current_hash)
        
    def _try_load_index(self, current_hash: str) -> bool:
        """
        尝试加载保存的索引和metadata
        """
        try:
            # 加载metadata
            with open(self.metadata_file, 'rb') as f:
                saved_data = pickle.load(f)
                saved_hash = saved_data['hash']
                
                # 检查hash是否匹配
                if saved_hash != current_hash:
                    return False
                    
                self.metadata = saved_data['metadata']
            
            # 加载向量索引
            self.index = faiss.read_index(self.vectors_file)
            return True
            
        except (FileNotFoundError, EOFError, pickle.PickleError):
            return False
            
    def _save_index(self, current_hash: str):
        """
        保存索引和metadata到文件
        """
        # 保存向量索引
        faiss.write_index(self.index, self.vectors_file)
        
        # 保存metadata和hash
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({
                'hash': current_hash,
                'metadata': self.metadata
            }, f)
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        搜索最相似的知识框架
        
        Args:
            query: 搜索主题
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Union[str, float]]]: 搜索结果列表，每个字典包含：
                - filename (str): 文件名
                - similarity_score (float): 相似度分数
        """
        if self.index is None:
            raise ValueError("请先调用build_index构建索引")
        
        # 限制top_k不能超过实际的文件数量
        top_k = min(top_k, len(self.metadata))
        
        # 获取查询向量
        query_vector = self._get_embedding(query)
        
        # 搜索最相似的向量
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            top_k
        )
        
        # 构建结果
        results = []
        for distance, file_idx in zip(distances[0], indices[0]):
            # 将欧氏距离转换为相似度分数（0-1之间）
            similarity = 1 / (1 + distance)
            results.append({
                'filename': self.metadata[file_idx]['filename'],
                'similarity_score': float(similarity)
            })
            
        return results

    def _get_new_files(self, summaries_dir: str) -> List[Path]:
        """获取新增的文件"""
        summaries_path = Path(summaries_dir)
        current_files = set(p.name for p in summaries_path.glob("*.txt"))
        indexed_files = set(meta['filename'] for meta in self.metadata.values())
        return [summaries_path / f for f in (current_files - indexed_files)]
    
    def update_index(self, summaries_dir: str):
        """增量更新索引"""
        if self.index is None:
            # 如果索引不存在，执行完整构建
            self.build_index(summaries_dir)
            return
            
        new_files = self._get_new_files(summaries_dir)
        if not new_files:
            return
            
        print(f"发现{len(new_files)}个新文件，正在更新索引...")
        
        # 处理新文件
        new_embeddings = []
        new_ids = []
        current_id = max(self.metadata.keys()) + 1 if self.metadata else 0
        
        for file_path in new_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            framework_part = content.split("第二部分：核心概念定义")[0].strip()
            embedding = self._get_embedding(framework_part)
            
            new_embeddings.append(embedding)
            new_ids.append(current_id)
            
            self.metadata[current_id] = {
                'filename': file_path.name,
                'summary_path': str(file_path),
                'article_path': str(file_path).replace('summaries', 'articles'),
                'last_updated': datetime.now().isoformat()
            }
            
            current_id += 1
        
        # 将新向量添加到现有索引
        if new_embeddings:
            self.index.add_with_ids(
                np.array(new_embeddings, dtype=np.float32),
                np.array(new_ids, dtype=np.int64)
            )
            
        # 保存更新后的索引和metadata
        current_hash = self._get_files_hash(summaries_dir)
        self._save_index(current_hash)