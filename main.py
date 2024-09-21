from utility import *


extracted_data = load_pdf("data/")

text_chunks = text_split(extracted_data)

embedding_model = download_hugging_face_embeddings()

llm = load_llm_model()

vs = vector_store(text_chunks,embedding_model)

retrivered_data = retriver(vs)

rg = rag_pipeline(llm,retrivered_data)

output(rg)