
# Clean Streamlit UI

import streamlit as st

from rag.utils import load_and_split_pdfs
from rag.pipeline import build_conversational_rag

st.title("Conversational RAG with PDF Uploads")
st.write("Upload PDFs and chat with their content")

session_id = st.text_input("Session ID", value="default")

if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    documents = load_and_split_pdfs(uploaded_files)

    if not documents:
        st.error("No text could be extracted from the document.")
        st.stop()

    conversational_rag_chain = build_conversational_rag(
        documents,
        st.session_state.store
    )

    user_question = st.text_input("Ask a question")

    if user_question:
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}}
        )


        st.subheader("Answer:")
        st.write(response['answer'])

        # Show Chat History
        session_history=st.session_state.store.get(session_id)
        if session_history:
            st.subheader("Chat History:")
            for msg in session_history.messages:
                role="User" if msg.type=="human" else "Assistant"
                st.write(f"**{role}:** {msg.content}")
