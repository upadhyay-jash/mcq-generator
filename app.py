# app.py

import streamlit as st
import tempfile
from utils import load_and_split_pdf, embed_documents, create_qa_chain

st.set_page_config(page_title="RAG-based MCQ Generator", layout="wide")
st.title("📘 RAG-based MCQ Generator")

uploaded_file = st.file_uploader("Upload a GenAI-related PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Step 1: Load & chunk the PDF
        docs = load_and_split_pdf(tmp_path)

        # Step 2: Embed & store in FAISS
        vector_store = embed_documents(docs)

        # Step 3: Create the RetrievalQA chain
        qa_chain = create_qa_chain(vector_store)

        # Step 4: Prompt the LLM
        result = qa_chain.run("Generate quiz questions from this document.")


        st.subheader("📝 Generated Questions")
        st.markdown(result)
