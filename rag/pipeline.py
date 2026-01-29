
# RAG + Retrieval + Chains

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.llm_utils import get_llm, get_embeddings


def build_conversational_rag(documents, session_store):
    embeddings = get_embeddings()
    llm = get_llm()

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Contextualize Question
    contextualize_q_system_prompt = (
        "Take the chat history and the user's latest question. "
        "Rewrite it if needed so it can be understood standalone."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Question-Answer Prompt
    system_prompt = (
        "You are a document-based question answering assistant. "
        "You MUST answer using ONLY the provided context. "
        "If the answer is not explicitly contained in the context, "
        "respond exactly with: 'I do not know based on the provided document.' "
        "Do NOT use prior knowledge. Do NOT guess.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_document_chain)    

    # Session Manage
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )
    return conversational_rag_chain
