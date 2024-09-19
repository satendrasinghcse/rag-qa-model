# type: ignore
import PyPDF2
import chromadb
from langchain.schema import Document
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate


load_dotenv()
key = os.getenv("groq_api_key")


def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def database_conn():
    client = chromadb.Client()
    collection = client.get_or_create_collection("knowledge_base2")


def doc_text(text, chunk_size=500):
    chunks =  [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return [Document(page_content=chunk) for chunk in chunks]


def embedding_model():
    embeddin_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=embeddin_model_name)


def model():
    model_name = "llama-3.1-70b-versatile"
    return ChatGroq(api_key=key,model=model_name)

def vector_store(documents,embedding_model):
    return Chroma.from_documents(documents,embedding_model)

def prompt():
    prompt_template="""
    You are an intelligent, helpful, and polite question-answering assistant designed to provide accurate, concise, and relevant answers based on the context provided. When responding to questions, ensure clarity, and be informative without overwhelming the user with unnecessary details. If the user asks for more information, provide additional clarification. Your role is to:

    Answer factual, domain-specific, or open-ended questions accurately.
    If uncertain about the answer, politely ask for clarification or additional context.
    Keep answers short and focused, unless the user requests more detail.
    In case of a subjective question, offer a balanced response.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return {"prompt": PROMPT}
 

def retriver_qa_chain(llm,vector_store,chain_type_kwargs):
    qa=RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs)
    return qa

def output(qa):
    while True:
        user_input=input(f"Ask your question:")
        result=qa({"query": user_input})
        print("Response : ", result["result"])
