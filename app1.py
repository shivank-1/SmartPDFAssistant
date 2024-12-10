import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.memory import ChatMessageHistory
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage
import pickle
import os
from dotenv import load_dotenv

# Load API keys from the environment
load_dotenv()
groqkey = os.getenv('groqkey')

# Downloading embeddings
def download_embeddings():
    embedding_path = "local_embeddings"

    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
    else:
        # Correct initialization of HuggingFaceEmbeddings
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)

    # Debug to ensure `embed_documents` method exists
    assert hasattr(embedding, 'embed_documents'), "Embedding instance lacks `embed_documents` method"
    return embedding

# Streamlit setup
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their contents")

# Input your Groq API key
api_key = groqkey

# Check if API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-70b-versatile", temperature=0.8)

    # Chat interface
    session_id = st.text_input("Session ID", value="default_session")

    # Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose A PDF File", type="pdf", accept_multiple_files=True)

    # Process the uploaded PDF
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"  # Temporary variable to store the contents of the original PDF
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        embedding = download_embeddings()
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
        retriever = vectorstore.as_retriever()

        # Contextualize system question prompt
        contextualize_system_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
                       "formulate a standalone question which can be understood without the chat history. "
                       "Do not answer the question, just reformulate it if needed and otherwise return it as it is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_system_q_prompt)

        # Answer prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
                       "If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Create the retrieval chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Chat history management
        def get_session_history(session: str) -> ChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # User input for questions
        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )

            st.write("Assistant Response:", response['answer'])
            st.write("Chat History:", session_history.messages)

    else:
        st.warning("Please upload a PDF file.")

else:
    st.warning("Please enter your Groq API Key.")
