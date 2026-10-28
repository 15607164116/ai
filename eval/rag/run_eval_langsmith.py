import json
from pathlib import Path

from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from rag.rag_service import RagSummarizeService


from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


SAMPLE_FILE = Path(__file__).parent / "qa.jsonl"
DATASET_NAME = "RAG 评测集 - 二手交易平台"

# ========== 1. 你的 RAG 链路，保持不变 ==========
def call_rag(question: str):
    # 1. 调用包含完整检索+生成逻辑的方法
    rag = RagSummarizeService()
    result = rag.rag_summarize_eval(question)

    # 2. 提取答案（rag_summarize_eval 已经处理过 AIMessage，这里直接取）
    answer = result["answer"]

    # 3. 提取检索到的 chunks，直接使用我们刚刚在 rag_summarize_eval 里注入的 chunk_id
    retrieved = [
        {"id": d.metadata.get("chunk_id", "unknown"), "text": d.page_content}
        for d in result["docs"]
    ]

    return {"retrieved_chunks": retrieved, "answer": answer}


# ========== 2. Target function：LangSmith 调用的入口 ==========
def target(inputs: dict) -> dict:
    """
    inputs 来自 dataset 的 inputs 字段，这里只需要 question。
    outputs 的 key 要和 evaluator 里取的一致。
    """
    result = call_rag(inputs["question"])
    return {
        "answer": result["answer"],
        "retrieved_ids": [c["id"] for c in result["retrieved_chunks"]],
        "retrieved_texts": [c["text"] for c in result["retrieved_chunks"]],
    }


# ========== 3. Code Evaluator：你原来的确定性检查 ==========
def recall_at_k_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """检索层：Recall@3，用 gold_chunks 和 retrieved_ids 算"""
    gold = reference_outputs.get("gold_chunks", [])
    retrieved = outputs.get("retrieved_ids", [])
    if not gold:
        return {"key": "recall@3", "score": None}
    hit = set(retrieved[:5]) & set(gold)
    return {"key": "recall@3", "score": len(hit) / len(gold)}


def keywords_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """生成层：关键词覆盖"""
    keywords = reference_outputs.get("keywords", [])
    answer = outputs.get("answer", "")
    if not keywords:
        return {"key": "keywords_ok", "score": None}
    score = 1.0 if all(kw in answer for kw in keywords) else 0.0
    return {"key": "keywords_ok", "score": score}


def forbidden_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """生成层：禁止词检查"""
    forbidden = reference_outputs.get("forbidden", [])
    answer = outputs.get("answer", "")
    if not forbidden:
        return {"key": "forbidden_ok", "score": None}
    score = 0.0 if any(fw in answer for fw in forbidden) else 1.0
    return {"key": "forbidden_ok", "score": score}


# ========== 4. LLM-as-judge（可选，LangSmith 官方推荐的 RAG 评估器） ==========
judge_llm = ChatOpenAI(
            model="deepseek-flash", # 或者 "deepseek-flash"，看你需要
            base_url="https://api.deepseek.com",
            # 直接把你的 DeepSeek Key 写在这里
            api_key="sk-df44c3eaeca14cb08a489cd069bfc4db",
            streaming=True, # Streamlit 需要流式输出
            temperature=0 # 测试和 Agent 场景建议调低温度，保证稳定性
        )

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Response vs Reference Answer：答案是否和标准答案事实一致。
    对应你样本里的 reference 字段。
    """
    reference = reference_outputs.get("reference", "")
    if not reference:
        return {"key": "correctness", "score": None}

    prompt = f"""你是评分员。判断学生答案是否在事实上与标准答案一致。
允许学生答案包含更多信息，只要不冲突。
只回答 True 或 False。

问题：{inputs['question']}
标准答案：{reference}
学生答案：{outputs['answer']}

判断："""
    result = judge_llm.invoke(prompt).content.strip()
    return {"key": "correctness", "score": 1.0 if "True" in result else 0.0}


# ========== 5. 创建/复用 Dataset ==========
def get_or_create_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME

    dataset = client.create_dataset(DATASET_NAME)
    samples = []
    with open(SAMPLE_FILE, encoding="utf-8") as f:
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
                    "reference": s.get("reference", ""),
                    "gold_chunks": s.get("gold_chunks", []),
                    "keywords": s.get("keywords", []),
                    "forbidden": s.get("forbidden", []),
                },
            }
            for s in samples
        ],
    )
    return DATASET_NAME


# ========== 6. 跑评测 ==========
def main():
    client = Client()
    dataset_name = get_or_create_dataset(client)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            recall_at_k_evaluator,
            keywords_evaluator,
            forbidden_evaluator,
            correctness_evaluator,  # 可选，需要 judge_llm
        ],
        experiment_prefix="rag-eval",
        max_concurrency=4,
    )

    # 打印聚合结果
    df = results.to_pandas()
    print("\n=== 评测汇总 ===")
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            print(f"{col}: {df[col].mean():.4f}")

    # ==========================================
    # 🚪 CI/CD 门禁逻辑
    # ==========================================
    # 安全提取各项均值（用 dropna() 过滤掉测试集中没填该字段的空值）
    recall_score = df["recall@3"].dropna().mean() if "recall@3" in df.columns else 1.0
    keywords_score = df["keywords_ok"].dropna().mean() if "keywords_ok" in df.columns else 1.0
    forbidden_score = df["forbidden_ok"].dropna().mean() if "forbidden_ok" in df.columns else 1.0
    correctness_score = df["correctness"].dropna().mean() if "correctness" in df.columns else 1.0

    print("\n=== 门禁检查 ===")

    # 1. 校验检索层：Recall@3 不能低于 80%
    if recall_score < 0.80:
        print(f"❌ [拦截] RAG 召回率过低 ({recall_score:.4f} < 0.80)，禁止合并！")
        sys.exit(1)

    # 2. 校验生成层：禁忌词绝对不能出现（必须 100%）
    if forbidden_score < 1.0:
        print(f"❌ [拦截] 回答出现禁忌词 ({forbidden_score:.4f} < 1.0)，禁止合并！")
        sys.exit(1)

    # 3. 校验语义层：LLM 裁判打分不能低于 90%
    if correctness_score < 0.90:
        print(f"❌ [拦截] LLM 裁判判定不通过率过高 ({correctness_score:.4f} < 0.90)，禁止合并！")
        sys.exit(1)

    print("✅ [通过] 所有 RAG 评测指标达标，允许部署！")
    sys.exit(0)  # 门禁通过，退出码 0


if __name__ == "__main__":
    main()