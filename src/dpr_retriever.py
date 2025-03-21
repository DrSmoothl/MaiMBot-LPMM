import numpy as np
from typing import List, Dict, Tuple
import os
from .config import global_config
from .llm_client import LLMClient
from .embedding_store import EmbeddingStore
from global_logger import logger
from .utils import get_md5
import json

def get_dpr_retriever(llm_client: LLMClient, raw_paragraphs: Dict[str, str] = None) -> 'DPRRetriever':
    """获取DPR检索器实例"""
    dpr_retriever = DPRRetriever(llm_client)
    logger.info("正在从文件加载DPR检索器")
    try:
        dpr_retriever.load()
    except Exception as e:
        logger.error("从文件加载DPR检索器时发生错误：{}".format(e))
        if raw_paragraphs is not None:
            # 如果加载失败且有原始段落数据，重新索引
            dpr_retriever.index_passages(raw_paragraphs)
            dpr_retriever.save()
    logger.info("DPR检索器加载完成")
    return dpr_retriever

class DPRRetriever:
    def __init__(self, llm_client: LLMClient):
        """初始化DPR检索器"""
        self.llm_client = llm_client
        self.embedding_store = EmbeddingStore(
            llm_client,
            "dpr",
            global_config["persistence"]["embedding_data_dir"]
        )
        self.passages: Dict[str, str] = {}  # hash_id -> passage
        self.passage_embeddings = None
        self.passage_hash_ids = []
        
    def index_passages(self, passages: Dict[str, str]):
        """索引段落"""
        logger.info("开始索引段落")
        # 为每个段落生成唯一的哈希ID
        nodes_dict = {}
        for doc_id, passage in passages.items():
            hash_id = f"dpr-{get_md5(passage)}"
            nodes_dict[hash_id] = {'content': passage, 'doc_id': doc_id}
            self.passages[hash_id] = passage
            
        # 获取所有哈希ID
        all_hash_ids = list(nodes_dict.keys())
        
        # 检查哪些段落需要新计算嵌入
        existing = self.embedding_store.store.keys()
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        
        if missing_ids:
            # 准备需要编码的段落
            passages_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
            # 批量计算嵌入
            missing_embeddings = self.llm_client.send_embedding_request(
                global_config["embedding"]["model"],
                passages_to_encode
            )
            # 保存新的嵌入
            self.embedding_store.batch_insert_strs(passages_to_encode)
            
        # 更新段落哈希ID列表
        self.passage_hash_ids = all_hash_ids
        # 获取所有段落的嵌入
        self.passage_embeddings = np.array([
            self.embedding_store.store[hash_id].embedding 
            for hash_id in self.passage_hash_ids
        ])
        
        logger.info(f"段落索引完成，共索引 {len(self.passage_hash_ids)} 个段落")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """检索相关段落，返回相关段落及其相似度分数"""
        # 计算查询文本的嵌入
        query_embedding = self.llm_client.send_embedding_request(
            global_config["embedding"]["model"],
            query
        )
        
        # 计算相似度
        similarities = np.dot(self.passage_embeddings, query_embedding)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 返回结果
        results = []
        for idx in top_indices:
            hash_id = self.passage_hash_ids[idx]
            similarity = similarities[idx]
            passage = self.passages[hash_id]
            results.append((passage, float(similarity)))
            
        return results
        
    def save(self):
        """保存DPR数据"""
        # 保存原始段落
        passages_file = os.path.join(
            global_config["persistence"]["embedding_data_dir"],
            "dpr_passages.json"
        )
        with open(passages_file, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False)
            
        # 保存段落哈希ID列表
        hash_ids_file = os.path.join(
            global_config["persistence"]["embedding_data_dir"],
            "dpr_hash_ids.json"
        )
        with open(hash_ids_file, "w", encoding="utf-8") as f:
            json.dump(self.passage_hash_ids, f, ensure_ascii=False)
            
    def load(self):
        """加载DPR数据"""
        # 加载原始段落
        passages_file = os.path.join(
            global_config["persistence"]["embedding_data_dir"],
            "dpr_passages.json"
        )
        if os.path.exists(passages_file):
            with open(passages_file, "r", encoding="utf-8") as f:
                self.passages = json.load(f)
                
        # 加载段落哈希ID列表
        hash_ids_file = os.path.join(
            global_config["persistence"]["embedding_data_dir"],
            "dpr_hash_ids.json"
        )
        if os.path.exists(hash_ids_file):
            with open(hash_ids_file, "r", encoding="utf-8") as f:
                self.passage_hash_ids = json.load(f)
                
        # 获取所有段落的嵌入
        if self.passage_hash_ids:
            self.passage_embeddings = np.array([
                self.embedding_store.store[hash_id].embedding 
                for hash_id in self.passage_hash_ids
            ]) 