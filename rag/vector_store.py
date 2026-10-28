from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf

from model.factory import embed_model

from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from rag.loader import parse_file

import os


def resolve_persist_directory() -> str:
    """解析向量库持久化目录，保证无论从哪个工作目录运行都指向同一个库。

    优先级：
    1. 环境变量 CHROMA_PERSIST_DIR（想完全自定义时用）
    2. 配置值 chroma_conf["persist_directory"]

    配置值会用 get_abs_path() 转成绝对路径。
    注意 os.path.join 的特性：若配置里写的本身就是绝对路径，会直接采用该绝对路径，
    所以 yml 里既支持写相对路径，也支持直接写绝对路径。
    """
    env_dir = os.environ.get("CHROMA_PERSIST_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    return get_abs_path(chroma_conf["persist_directory"])


class VectorStoreService:
    def __init__(self):
        # 固定向量库位置：绝对路径，避免不同工作目录下各自生成空库
        self.persist_directory = resolve_persist_directory()

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=self.persist_directory,
        )

        self._log_and_check()

        # md：先按标题切分成语义小节
        self.md_header_spliter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
            strip_headers=False,   # 保留标题文本，让分片自带上下文，利于语义检索
        )

        # 通用长度切分：txt 用它；md 按标题切完后，仅超长小节用它二次切
        # 换行优先于句号，避免切出"孤儿句号块"
        self.text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=["\n\n", "\n", "。", "，", " ", ""],
            length_function=len,
        )

    def _split_document(self, text: str, path: str) -> list[Document]:
        """按文件类型选择切分策略"""
        if path.endswith((".md", ".markdown")):
            chunks = self._split_markdown(text)
        else:
            # txt 没有 markdown 标题，直接按长度切
            chunks = self.text_spliter.create_documents([text])

        # 补充来源，供日志与溯源使用
        for chunk in chunks:
            chunk.metadata.setdefault("source", path)

        return chunks

    def _split_markdown(self, text: str) -> list[Document]:
        """md 两段式切分：先按标题切，再仅对超长小节二次切"""
        header_documents: list[Document] = self.md_header_spliter.split_text(text)

        result: list[Document] = []
        for doc in header_documents:
            # 短小节本身就是语义完整单元，不再二次切分，避免打碎刚拼好的语义
            if len(doc.page_content) > chroma_conf["chunk_size"]:
                result.extend(self.text_spliter.split_documents([doc]))
            else:
                result.append(doc)

        return result

    def _log_and_check(self):
        """打印实际使用的向量库位置，并在库为空时给出醒目告警"""
        logger.info(f"[VectorStoreService]向量库固定路径：{self.persist_directory}")

        try:
            count = self.vector_store._collection.count()
        except Exception as e:      # 集合尚未创建等异常情况
            logger.warning(f"[VectorStoreService]无法读取向量库条目数：{e}")
            return

        logger.info(f"[VectorStoreService]当前向量库条目数：{count}")

        if count == 0:
            logger.error(
                "[VectorStoreService]向量库为空，检索必定命中不到数据！"
                "请先在项目根目录执行灌库："
                "python -c \"from rag.vector_store import VectorStoreService; VectorStoreService().load_document()\""
                "（务必在【项目根目录】执行，否则会在当前工作目录生成新的空库）"
            )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        :return: None
        """

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:

            try:
                text: str = parse_file(path)

                if not text.strip():
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self._split_document(text, path)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


