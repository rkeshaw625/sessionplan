import fitz  # PyMuPDF
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Memory storage for chat
memory = ConversationBufferMemory()

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def create_vector_store(text_chunks, api_key):
    """Creates a FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_pdf_themes(llm, text_chunks):
    """Extracts 5 major themes from the PDF using the LLM."""
    prompt = SystemMessage(
        content="You are an AI trained to analyze documents and extract key themes. Provide 5 major themes from the following text:"
    )
    
    themes_response = llm.invoke([prompt, HumanMessage(content="\n".join(text_chunks[:5]))])
    themes = themes_response.content.split("\n")[:5]
    
    return [theme.strip("- ") for theme in themes if theme.strip()]

def generate_learning_session(llm, theme, duration):
    """Generates an experiential learning session for a selected theme."""
    prompt = SystemMessage(
        content=f"You are an expert in experiential learning design. Create a {duration}-minute learning session on '{theme}'."
    )
    
    session_response = llm.invoke([prompt])
    return session_response.content

def answer_question_with_memory(llm, vector_store, question):
    """Retrieves relevant chunks using FAISS and answers the question while maintaining memory."""
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, memory=memory)
    
    response = qa_chain.run(question)
    return response
