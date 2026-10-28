from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.logger_handler import logger

# 历史消息保留上限，超出后丢弃最早的消息，避免多轮对话后 token 无限膨胀
MAX_HISTORY_MESSAGES = 40


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        # 多轮对话记忆：保存完整的消息对象（含 tool_calls 与 ToolMessage），而非纯文本
        self.history: list = []
        self.last_tools_called = []
        self.last_retrieved_docs = []

    def execute_stream(self, query: str):
        # 把本轮用户提问追加进历史，作为本次 Agent 的输入
        self.history.append(HumanMessage(content=query))

        input_dict = {"messages": self.history}

        final_messages = self.history
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        # 每轮重新构造，保证 report 标记不会从上一轮泄漏到本轮
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            final_messages = chunk["messages"]
            latest_message = final_messages[-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

        # 流结束后回写完整历史（包含本轮的工具调用及其结果）
        self.history = self._trim_history(final_messages)
        logger.info(f"[ReactAgent]本轮结束，当前上下文消息数：{len(self.history)}")

    @staticmethod
    def _trim_history(messages: list) -> list:
        """超出上限时丢弃最早的消息；若截断后首条是孤儿 ToolMessage 则一并丢弃，
        避免 AIMessage(tool_calls) 与其 ToolMessage 被拆散，导致模型端校验报错"""
        if len(messages) <= MAX_HISTORY_MESSAGES:
            return messages

        trimmed = list(messages[-MAX_HISTORY_MESSAGES:])
        while trimmed and isinstance(trimmed[0], ToolMessage):
            trimmed.pop(0)

        return trimmed

    def clear_history(self):
        """清空多轮对话记忆"""
        self.history = []
        logger.info("[ReactAgent]对话历史已清空")



if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
