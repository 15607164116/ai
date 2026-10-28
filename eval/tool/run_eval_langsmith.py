import json
from pathlib import Path
from langsmith import Client, evaluate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# ==========================================
# 1. 自定义 Callback 拦截器 (用于抓取工具调用)
# ==========================================
class ToolTrackerCallback(BaseCallbackHandler):
    def __init__(self):
        self.called_tools = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """当 Agent 开始调用某个工具时触发"""
        if serialized and "name" in serialized:
            self.called_tools.append(serialized["name"])


# ==========================================
# 2. Target Function：适配你的 ReactAgent
# ==========================================
def target(inputs: dict) -> dict:
    # 动态导入，避免每次运行加载过慢
    from agent.react_agent import ReactAgent

    agent = ReactAgent()
    tracker = ToolTrackerCallback()
    question = inputs["question"]

    # 1. 注入当前问题
    agent.history.append(HumanMessage(content=question))
    input_dict = {"messages": agent.history}

    final_answer = ""
    called_tools = []

    # 2. 直接调用底层 agent 的 stream 跑完流程
    # 注意：这里用 agent.agent.stream 而不是 agent.execute_stream，
    # 这样我们既能挂载 callbacks，又能自己控制解析逻辑
    for chunk in agent.agent.stream(
            input_dict,
            stream_mode="values",
            context={"report": False},
            config={"callbacks": [tracker]}
    ):
        latest_message = chunk["messages"][-1]

        # 如果当前节点是 AI 决定调用工具
        if isinstance(latest_message, AIMessage) and latest_message.tool_calls:
            for tc in latest_message.tool_calls:
                called_tools.append(tc["name"])

        # 如果当前节点是 AI 的最终回复（没有工具调用了）
        if isinstance(latest_message, AIMessage) and not latest_message.tool_calls:
            if latest_message.content:
                final_answer = latest_message.content

    # 如果 Callback 抓到了工具调用，优先使用 Callback 的（更稳）
    if tracker.called_tools:
        called_tools = tracker.called_tools

    # 去重并保持顺序
    called_tools = list(dict.fromkeys(called_tools))

    return {
        "answer": str(final_answer),
        "called_tools": called_tools
    }


# ==========================================
# 3. Evaluators：评测工具选择
# ==========================================
def tool_selection_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """工具召回率：期望调用的工具是否都被调用了"""
    expected = set(reference_outputs.get("expected_tools", []))
    called = set(outputs.get("called_tools", []))

    if not expected:
        # 如果期望不调工具（比如闲聊），且确实没调，得1分
        return {"key": "tool_selection_acc", "score": 1.0 if not called else 0.0}

    hit = expected & called
    score = len(hit) / len(expected)
    return {"key": "tool_selection_acc", "score": score}


def forbidden_tool_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """禁止工具拦截：有没有乱调不该调的"""
    forbidden = set(reference_outputs.get("forbidden_tools", []))
    called = set(outputs.get("called_tools", []))

    if not forbidden:
        return {"key": "forbidden_tool_ok", "score": 1.0}

    score = 0.0 if (forbidden & called) else 1.0
    return {"key": "forbidden_tool_ok", "score": score}


# ==========================================
# 4. 准备 LangSmith 数据集
# ==========================================
DATASET_NAME = "工具调用评测集 - ReAct Agent"
TOOL_FILE = Path(__file__).parent / "qa.jsonl"


def get_or_create_tool_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME

    dataset = client.create_dataset(DATASET_NAME)
    samples = []
    with open(TOOL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": s["question"]},
                "outputs": {
                    "expected_tools": s.get("expected_tools", []),
                    "forbidden_tools": s.get("forbidden_tools", [])
                },
            }
            for s in samples
        ],
    )
    return DATASET_NAME


# ==========================================
# 5. 运行
# ==========================================
def main():
    client = Client()
    dataset_name = get_or_create_tool_dataset(client)

    print(f"正在跑数据集: {dataset_name} ...")

    # 跑评测
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            tool_selection_evaluator,
            forbidden_tool_evaluator,
        ],
        experiment_prefix="tool-eval",
        max_concurrency=1,  # 保持为1，避免多个请求同时打满 API 或状态错乱
    )

    # 打印汇总结果
    df = results.to_pandas()
    print("\n=== 工具评测汇总 ===")
    for col in df.columns:
        if "acc" in col or "ok" in col:
            print(f"{col}: {df[col].mean():.4f}")

    # ==========================================
    # 🚪 CI/CD 门禁逻辑
    # ==========================================
    # 安全提取各项均值（用 dropna 排除空值）
    tool_acc = df["tool_selection_acc"].dropna().mean() if "tool_selection_acc" in df.columns else 1.0
    forbidden_ok = df["forbidden_tool_ok"].dropna().mean() if "forbidden_tool_ok" in df.columns else 1.0

    print("\n=== 门禁检查 ===")

    # 1. 校验工具选择准确率：不能低于 90%
    if tool_acc < 0.90:
        print(f"❌ [拦截] 工具选择准确率过低 ({tool_acc:.4f} < 0.90)，禁止合并！")
        sys.exit(1)

    # 2. 校验禁忌工具：这是红线，必须 100% 不能触发
    if forbidden_ok < 1.0:
        print(f"❌ [拦截] Agent 出现了乱调工具的情况 ({forbidden_ok:.4f} < 1.0)，禁止合并！")
        sys.exit(1)

    print("✅ [通过] 工具调用评测指标全部达标，允许部署！")
    sys.exit(0)  # 正常退出，通行


if __name__ == "__main__":
    main()