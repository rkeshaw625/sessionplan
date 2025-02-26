import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.llms import GooglePalm
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import utils  # Custom helper functions

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄🔍 RAG Chatbot with Memory & Vector Search")

# Sidebar for API Key and LLM Selection
st.sidebar.header("🔑 API Key Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI/Gemini API key", type="password")
llm_choice = st.sidebar.selectbox("Select LLM Provider", ["OpenAI", "Gemini"])

if not api_key:
    st.warning("Please enter an API key to proceed.")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        pdf_text = utils.extract_text_from_pdf(uploaded_file)
    
    # Chunking and Vector Search Initialization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(pdf_text)

    with st.spinner("Embedding document into vector store..."):
        vector_store = utils.create_vector_store(text_chunks, api_key)

    # LLM Selection
    llm = ChatOpenAI(api_key=api_key, model="gpt-4") if llm_choice == "OpenAI" else GooglePalm(api_key=api_key)
    
    # Generate 5 major themes
    with st.spinner("Analyzing document..."):
        themes = utils.get_pdf_themes(llm, text_chunks)
    
    st.subheader("📌 Key Themes Identified:")
    selected_theme = st.radio("Select a theme to generate a learning session:", themes)

    # User inputs session duration
    duration = st.slider("Set session duration (minutes)", 10, 120, 30)

    if st.button("Generate Experiential Learning Session"):
        with st.spinner("Generating learning session..."):
            session_plan = utils.generate_learning_session(llm, selected_theme, duration)
        
        st.subheader("🎓 Experiential Learning Session")
        st.markdown(session_plan)

    # Interactive Chat Mode
    st.subheader("💬 Ask Questions About the Document")
    chat_input = st.text_input("Type your question here:")

    if chat_input:
        with st.spinner("Searching document and generating response..."):
            response = utils.answer_question_with_memory(llm, vector_store, chat_input)
        
        st.markdown(f"**🤖 Chatbot:** {response}")
