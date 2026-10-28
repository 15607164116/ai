from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


def _format_message(msg, index: int) -> str:
    """把单条消息格式化为易读的日志行"""
    role = getattr(msg, "role", type(msg).__name__)

    # content 通常是 str，多模态消息下也可能是 list
    content = msg.content
    if isinstance(content, str):
        text = content.strip()
    else:
        text = str(content) if content is not None else ""

    line = f"  [{index}] {type(msg).__name__}(role={role}): {text}"

    # AIMessage 上可能挂着工具调用，而 content 恰好为空
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        calls = ", ".join(
            f"{tc.get('name', '?')}({tc.get('args', {})})" for tc in tool_calls
        )
        line += f"\n        └─ tool_calls: {calls}"

    # ToolMessage 会回带它响应的那条 tool_call 的 id
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        line += f"\n        └─ tool_call_id: {tool_call_id}"

    return line


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    messages = state['messages']

    logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息。")

    logger.info("[log_before_model]本次调用模型的完整消息列表：")
    for index, msg in enumerate(messages):
        logger.info(_format_message(msg, index))

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
