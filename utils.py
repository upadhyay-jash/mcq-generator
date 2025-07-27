# utils.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import config

def load_and_split_pdf(pdf_path):
    """Load and split the PDF into chunks"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    return docs

def embed_documents(docs):
    embeddings = AzureOpenAIEmbeddings(
    azure_deployment=config.AZURE_DEPLOYMENT_NAME_EMBEDDING,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,            # ✅ use this now
    openai_api_key=config.AZURE_OPENAI_API_KEY,
    openai_api_version=config.AZURE_API_VERSION,
    chunk_size=1000
)
    return FAISS.from_documents(docs, embeddings)


def create_qa_chain(vector_store):
    prompt_template = """
You are a smart quiz generator. Based on the following context from a GenAI-related PDF, generate:
1. 5 Multiple Choice Questions (MCQs) with 4 options each, and mark the correct answer.
2. 5 Short Answer Questions.

Context:
{context}

Questions:
"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context"]
    )

    llm = AzureChatOpenAI(
        deployment_name=config.AZURE_DEPLOYMENT_NAME_CHAT,
        openai_api_key=config.AZURE_OPENAI_API_KEY,
        openai_api_base=config.AZURE_OPENAI_ENDPOINT,
        openai_api_version=config.AZURE_API_VERSION,
        temperature=0.7
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )

    return chain
