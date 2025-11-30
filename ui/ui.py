import streamlit as st
from app import CV_RAG_Assistant

st.title("CardioRAG: Diagnosis & Treatment Assistant")
st.caption("Powered by your 400-page cardiovascular pharmacotherapy PDF")

assistant = CV_RAG_Assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about diagnosis, drugs, or guidelines..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = assistant.ask(prompt)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})