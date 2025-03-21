
import json
from typing import Dict, List
import tqdm
import time
import os

from .config import global_config
from .llm_client import LLMClient
from . import prompt_template
from global_logger import logger



def triple_llm_filter(llm_client: LLMClient, question: str, triples: List[tuple]) -> List[tuple]:
    """使用LLM过滤三元组"""
    # 构建提示词
    triple_filter_context = prompt_template.build_llm_filter_context(question, triples)
    
    # 发送请求
    response = llm_client.send_chat_request(
        global_config["triple_filter"]["llm"]["model"],
        triple_filter_context
    )
    
    # 解析JSON响应
    try:
        if "</think>" in response:
            response = response.split("</think>")[1]
        if "{" in response:
            response = response[response.index("{"):]
        if "}" in response:
            response = response[:response.rindex("}") + 1]
            
        result = json.loads(response)
        filtered_triples = result["filtered_triples"]
        
        # 验证三元组格式        
        for triple in filtered_triples:
            if len(triple) != 3 or (
                triple[0] is None or triple[1] is None or triple[2] is None
            ):
                raise Exception("三元组格式错误")
        return filtered_triples
    except Exception as e:
        logger.error(f"三元组过滤失败: {e}")
        return triples
    
def process_triple_filter(
    llm_client: LLMClient,
    question: str,
    triples_data: dict,
):
    """处理三元组过滤任务"""
    # 该任务需要读取过滤后的三元组结果，所以需要读取filtered_triples.json文件
    logger.info("正在读取过滤后的三元组文件")
    filtered_file = global_config["persistence"].get("filtered_triples_path", "data/filtered_triples.json")
    filtered_triples_json = None
    
    if os.path.exists(filtered_file) is True:
        with open(filtered_file, "r", encoding="utf-8") as f:
            try:
                filtered_triples_json = json.loads(f.read())
            except json.JSONDecodeError:
                filtered_triples_json = None

    if filtered_triples_json is None:
        logger.error("过滤后的三元组文件为空/不存在/格式错误")
        # 构建过滤后的三元组
        logger.info("开始执行三元组过滤任务")
        filtered_triples_json = {}
        
        # 合并所有三元组
        all_triples = []
        for triples in triples_data.values():
            all_triples.extend(triples)
            
        try_count = 0
        while try_count < 3:
            try:
                filtered_triples = triple_llm_filter(
                    llm_client,
                    question,
                    all_triples
                )
                filtered_triples_json["filtered_triples"] = filtered_triples
                break
            except Exception as e:
                logger.error("三元组过滤任务失败，原因：{}".format(e))
                logger.warning("于5s后重试三元组过滤请求...")
                try_count += 1
                time.sleep(5)
                
            if try_count == 3:
                logger.error("三元组过滤任务失败，已重试3次")
                return []
            

        # 将过滤后的三元组写回文件
        with open(filtered_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(filtered_triples_json, ensure_ascii=False))

        logger.info("三元组过滤任务完成")
        
    else:
        logger.info("过滤后的三元组文件读取成功")

    return filtered_triples_json.get("filtered_triples", []) 





