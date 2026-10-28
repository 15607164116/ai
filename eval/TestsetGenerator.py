import os
from openai import OpenAI
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "data")
# 1. 加载文档
loader = DirectoryLoader(
    data_path,
    glob="**/*.md",
    loader_cls=TextLoader,          # 强制用纯文本方式读取
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
for doc in documents:
    doc.metadata['filename'] = doc.metadata.get('source', 'unknown')

# 2. 用原生 client 初始化（DeepSeek 兼容 OpenAI 接口）
client = OpenAI(
    api_key="sk-df44c3eaeca14cb08a489cd069bfc4db",
    base_url="https://api.deepseek.com"
)

# 3. 创建 generator
generator = TestsetGenerator(
    llm=llm_factory("deepseek-chat", client=client),
    embedding_model=OpenAIEmbeddings(client=client)
)

# 4. 生成（注意参数名是 testset_size，不是 test_size）
testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=10  # 先用 10 条跑通
)

# 5. 导出
testset.to_pandas().to_json("rag_testset.json", orient="records", force_ascii=False)
print("✅ 生成完成")