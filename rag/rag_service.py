
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger

# 单条参考资料在日志中打印的最大字符数，超出部分截断，避免日志爆炸
REFERENCE_LOG_MAX_LEN = 300


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)

        logger.info(f"[rag_summarize]用户提问：{query}")
        logger.info(f"[rag_summarize]向量检索命中 {len(context_docs)} 条参考资料")

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1

            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            location = f"{source}" + (f" 第{page + 1}页" if page is not None else "")

            content = doc.page_content.strip()
            if len(content) > REFERENCE_LOG_MAX_LEN:
                shown = content[:REFERENCE_LOG_MAX_LEN] + f"...(截断，共{len(content)}字符)"
            else:
                shown = content

            logger.info(f"[rag_summarize]【参考资料{counter}】来源：{location}")
            logger.info(f"[rag_summarize]【参考资料{counter}】内容：{shown}")

            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        if not context_docs:
            logger.warning(f"[rag_summarize]未检索到任何参考资料，模型将缺少依据")

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

    def rag_summarize_eval(self, query: str):

        # 1. 检索文档（注意：如果你的 retriever_docs 不支持 k 参数，需要去那边改一下）
        context_docs = self.retriever_docs(query)

        logger.info(f"[rag_summarize]用户提问：{query}")
        logger.info(f"[rag_summarize]向量检索命中 {len(context_docs)} 条参考资料")

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1

            # ==========================================
            # 核心修改点：利用你库里的 h2 作为稳定的 chunk_id
            # ==========================================
            if "chunk_id" not in doc.metadata:
                # 如果 metadata 里有 h2，就用 h2 (如"如何发布商品")，否则兜底用 counter
                doc.metadata["chunk_id"] = doc.metadata.get("h2", f"chunk_{counter}")

            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            location = f"{source}" + (f" 第{page + 1}页" if page is not None else "")

            content = doc.page_content.strip()
            if len(content) > REFERENCE_LOG_MAX_LEN:
                shown = content[:REFERENCE_LOG_MAX_LEN] + f"...(截断，共{len(content)}字符)"
            else:
                shown = content

            logger.info(f"[rag_summarize]【参考资料{counter}】来源：{location}")
            logger.info(f"[rag_summarize]【参考资料{counter}】内容：{shown}")

            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        if not context_docs:
            logger.warning(f"[rag_summarize]未检索到任何参考资料，模型将缺少依据")

        # 调用模型生成答案
        answer = self.chain.invoke({"input": query, "context": context})

        # 兼容 AIMessage 对象，直接提取纯文本
        if hasattr(answer, "content"):
            answer = answer.content

        return {"answer": answer, "docs": context_docs}



if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("如何发布商品"))
