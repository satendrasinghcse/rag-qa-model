import os
from dotenv import load_dotenv 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()
groq_key = os.getenv("groq_api_key")

def load_pdf(data_dir):
    """Load PDF documents from a specified directory."""
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """Split extracted documents into manageable text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    """Download and return Hugging Face embeddings model."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def load_llm_model():
    """Load and return the large language model (LLM)."""
    model_name = "llama-3.1-70b-versatile"
    return ChatGroq(api_key=groq_key, model=model_name)

def vector_store(text_chunks, embedding_model):
    """Create and return a vector store from the text chunks and embedding model."""
    return Chroma.from_documents(text_chunks, embedding_model)

def retriver(vector_store):
    """Return a retriever from the vector store."""
    return vector_store.as_retriever()

def rag_pipeline(llm, retriever):
    """Create and return a Retrieval-Augmented Generation (RAG) pipeline."""
    rag_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return rag_qa

def output(rag_qa):
    """Continuously take user input and invoke the RAG pipeline for responses."""
    while True:
        user_input = input("Input Prompt: ")
        result = rag_qa.invoke(user_input)
        print("Response: ", result["result"])
