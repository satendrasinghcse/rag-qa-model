from utility import *
import streamlit as st # type: ignore
import warnings 

 
warnings.filterwarnings('ignore') 




def training(pdf):
        st.write("Training has started...")
        # Call your training function here
        #train_model(text)  # Replace with your actual training function
        print("text extraction start...")
        text = extract_text_from_pdf(pdf)
        print("text extraction completed.")

        database_conn()
        print("database connected")
        document = doc_text(text)

        embedd_model = embedding_model()
        print("emebedding model lodded")
        llm = model()
        print("llm model lodded")
        promts = prompt()
        vs = vector_store(document,embedd_model)
        qas = retriver_qa_chain(llm,vs,promts)
        print("model traing completed.")
        return qas,text


# File upload
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
if uploaded_file:
      qa,tx=training(uploaded_file)
      query = st.text_input("Ask a question:")

      if query:
           answer = qa(query, tx)
           st.write(f"Answer: {answer["result"]}")




    



