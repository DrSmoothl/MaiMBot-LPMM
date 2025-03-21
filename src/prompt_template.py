from .llm_client import LLMMessage

entity_extract_system_prompt = """你是一个性能优异的实体提取系统。请从段落中提取出所有实体，并以JSON列表的形式输出。

输出格式示例：
{
    “named_entities”: ["实体A", "实体B", "实体C"]
}

请注意以下要求：
- 尽可能多的提取出段落中的全部实体；
- 将代词转化为对应的实体命名，以避免指代不清。"""


def build_entity_extract_context(paragraph: str) -> str:
    messages = []
    messages.append(LLMMessage("system", entity_extract_system_prompt).to_dict())
    messages.append(LLMMessage("user", f"""段落：\n```\n{paragraph}```""").to_dict())
    return messages

llm_filter_system_prompt = """你是一个性能优异的三元组过滤系统。你的任务是根据给定的问题和三元组列表，判断每个三元组与问题的相关性。

请使用JSON回复，输出过滤后的三元组列表。

输出格式示例：
{
    "filtered_triples": [
        ["实体A", "关系", "实体B"],
        ["实体C", "关系", "实体D"]
    ]
}

请注意以下要求：
- 仅保留与问题语义相关的三元组
- 三元组的相关性判断应基于问题的意图和主题
- 如果三元组包含的实体或关系与问题中提到的概念有关,应当保留
- 如果三元组能够帮助回答问题或提供相关背景信息,应当保留"""


def build_llm_filter_context(question: str, triples: list) -> str:
    messages = []
    messages.append(LLMMessage("system", llm_filter_system_prompt).to_dict())
    messages.append(
        LLMMessage(
            "user",
            f"""问题：\n```\n{question}```\n\n三元组列表：\n```\n{triples}```"""
        ).to_dict()
    )
    return messages


rdf_triple_extract_system_prompt = """你是一个性能优异的RDF（资源描述框架，由节点和边组成，节点表示实体/资源、属性，边则表示了实体和实体之间的关系以及实体和属性的关系。）构造系统。你的任务是根据给定的段落和实体列表构建RDF图。

请使用JSON回复，使用三元组的JSON列表输出RDF图中的关系（每个三元组代表一个关系）。

输出格式示例：
{
    "triples":[
        ["某实体","关系","某属性"],
        ["某实体","关系","某实体"],
        ["某资源","关系","某属性"]
    ]
}

请注意以下要求：
- 每个三元组应包含每个段落的实体命名列表中的至少一个命名实体，但最好是两个。
- 将代词（如“你”、“我”、“他”、“她”、“它”等）清楚地解析为其具体名称以保持清晰度。"""


def build_rdf_triple_extract_context(paragraph: str, entities: str) -> str:
    messages = []
    messages.append(LLMMessage("system", rdf_triple_extract_system_prompt).to_dict())
    messages.append(
        LLMMessage(
            "user", f"""段落：\n```\n{paragraph}```\n\n实体列表：\n```\n{entities}```"""
        ).to_dict()
    )
    return messages


qa_system_prompt = """你是一个性能优异的QA系统。请根据给定的问题，从给定的知识库中筛选与问题有关的要点，并作出回答。"""


def build_qa_context(question: str, knowledge: list[(str, str, str)]) -> str:
    messages = []
    messages.append(LLMMessage("system", qa_system_prompt).to_dict())
    messages.append(
        LLMMessage(
            "user", f"问题：\n{question}\n\n你所掌握的知识要点：\n{knowledge}"
        ).to_dict()
    )
    return messages
