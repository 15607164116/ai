import streamlit as st
from agent.react_agent import ReactAgent

st.set_page_config(page_title="智扫通 · 智能客服", page_icon="🤖")
st.title("🤖 智扫通机器人智能客服")
st.caption("基于 LangChain ReAct Agent + RAG 检索增强")
st.divider()

from dotenv import load_dotenv
load_dotenv()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.markdown("### 对话控制")
    if st.button("清空对话", use_container_width=True):
        st.session_state["agent"].clear_history()
        st.session_state["messages"] = []
        st.rerun()
    st.caption(f"当前上下文消息数：{len(st.session_state['agent'].history)}")
    st.caption("Agent 会记住本页面的多轮对话，换话题时可点击上方清空。")

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    response_messages: list[str] = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def stream_generator(generator, cache_list):
            """逐字流式输出，同时缓存完整响应"""
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    yield char

        st.chat_message("assistant").write_stream(stream_generator(res_stream, response_messages))
        st.session_state["messages"].append({"role": "assistant", "content": "".join(response_messages)})
        st.rerun()
