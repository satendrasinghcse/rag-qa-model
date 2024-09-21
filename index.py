import streamlit as st
import os
from utility import *

# Set the page configuration
st.set_page_config(page_title="QA System", layout="centered")

# Page title and description
st.title("📄 Question-Answering System")
st.markdown("""
Welcome to the **QA System**! This tool allows you to upload multiple PDF documents, 
train a language model on their content, and then ask questions about the documents to get accurate answers.
""")

# Load the embedding model and language model
embedding_model = download_hugging_face_embeddings()
llm = load_llm_model()

# Directory to save uploaded PDFs
UPLOAD_FOLDER = 'data'

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# PDF file uploader widget for multiple files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Handle uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the uploaded file name
        st.write(f"Uploaded file: **{uploaded_file.name}**")

        # Save the uploaded file to the specified directory
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File saved successfully: `{save_path}`")

# Define the training function
def training():
    # Load and process all PDF documents
    extracted_data = load_pdf(UPLOAD_FOLDER)
    text_chunks = text_split(extracted_data)
    # Create and return the vector store
    return vector_store(text_chunks, embedding_model)

# Initialize session state for vector store
if 'vs' not in st.session_state:
    st.session_state.vs = None

# Button to start training the model
if st.button("Start Training", key="start_training", type="primary"):
    with st.spinner("Training in progress... Please wait."):
        # Call the training function and store the result in session state
        st.session_state.vs = training()
        st.success("Model trained successfully!")

# Input for user query
query = st.text_input("Ask a question based on the documents")

# Handle the query and display the answer
if query:
    with st.spinner("Retrieving answer..."):
        # Retrieve the relevant data using the vector store
        retrieved_data = retriver(st.session_state.vs)
        
        # Create a pipeline for generating the answer
        rg = rag_pipeline(llm, retrieved_data)
        
        # Generate the answer based on the query
        answer = rg(query)
        
        # Display the answer
        st.write(f"**Answer:** {answer['result']}")

# Footer section
st.markdown("---")  # A horizontal line for separation
st.markdown("""
*Created by Satendra Singh | Powered by llama3*  
""")
