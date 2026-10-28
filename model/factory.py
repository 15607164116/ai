from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self):
        return ChatOpenAI(
            model="deepseek-flash", # 或者 "deepseek-flash"，看你需要
            base_url="https://api.deepseek.com",
            # 直接把你的 DeepSeek Key 写在这里
            api_key="sk-df44c3eaeca14cb08a489cd069bfc4db",
            streaming=True, # Streamlit 需要流式输出
            temperature=0 # 测试和 Agent 场景建议调低温度，保证稳定性
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        return DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key="sk-ws-H.EXEYEDY.zo89.MEUCIQCbqcjlpKmCqnUUhley1x8boxtNncxOCGvawu2KNJJeZAIgafLJ-SLyULnZGffqRa5rI8s3FRj6QFm7Z0fSWFlB3Kg"
        )


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()
