import streamlit as st
from utils import create_index_documents, graph_rag
from langchain.schema import AIMessage
import uuid

st.set_page_config(
    page_title="IT Bot",
    page_icon="📊",
    layout="wide"
)

left_col, right_col = st.columns([5, 1])
with left_col:
    st.title("🔍IT Operations Bot")

with right_col:
    col1, col2 = st.columns([1, 1])  # two equal-width columns

    with col1:
        if st.button("Create Index"):
            with st.spinner("Indexing documents..."):
                create_index_documents()
            st.toast("Index is created")

    with col2:
        if st.button("Clear", type="tertiary"):
            st.session_state.messages = []
            st.toast("Chat is cleared")
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = uuid.uuid4()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        placeholder= st.empty()
        config = {"configurable": {"thread_id": st.session_state.chat_id}}
        full_response = ""
        for step in graph_rag.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="values",
            config=config,
        ):
            msg = step["messages"][-1]

            if isinstance(msg, AIMessage):
                full_response += msg.content
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})